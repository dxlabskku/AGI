import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchvision
from torchvision.io import read_video


if hasattr(torchvision, "set_video_backend"):
    try:
        torchvision.set_video_backend("pyav")
    except Exception:
        pass  
import os
import  argparse, warnings
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torchvision.io import read_video
from torchcodec.decoders import VideoDecoder

from encoder2 import (
    VideoSwinBackbone, VideoSwinCfg,
    AudioCLAPWindowed, CLAPAudioCfg,
)


# ---------- 유틸 ----------
CODE2IDX = {"A":0, "B1":1, "B2":2, "B4":3, "B5":4, "B6":5, "G":6}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def parse_label_from_id(file_id: str):
    tail = file_id.split("_")[-1]
    codes = tail.split("-")
    binary = 1 if len(codes) > 1 else 0
    multilabel = [CODE2IDX[c] for c in codes if c != "0"]
    return int(binary), multilabel

def read_list_ids(list_path: str) -> List[str]:
    ids = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()[0]
            s = s.rstrip(".mp4")
            if "/" in s:
                s = s.split("/")[-1]
            ids.append(s)
    return ids

def resample_mono(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if wav.dim() == 2 and wav.shape[1] > 1:
        wav = wav.mean(dim=1)  # (S,)
    elif wav.dim() == 2:
        wav = wav[:,0]
    elif wav.dim() == 1:
        pass
    else:
        wav = wav.view(-1)
    if orig_sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    return wav

# ---------- 논문 파라미터 ----------
FPS_FIXED = 24.0
T_FRAMES  = 16
TS_SEC    = T_FRAMES / FPS_FIXED           # 0.6666667 s
AUDIO_WIN = 0.96                           # 960 ms
AUDIO_HOP = TS_SEC                         # 스니펫 길이만큼 hop, 끝 시간 정렬
def read_video_safe(path):

    try:
        v, a, info = read_video(path, pts_unit="sec")
        return v, a, info
    except Exception:
        pass

    try:
        import decord as de
        de.bridge.set_bridge('torch')
        vr = de.VideoReader(path)
        v = vr.get_batch(list(range(len(vr))))  # [T,H,W,3] uint8
        info = {"video_fps": float(vr.get_avg_fps()), "video_frames": int(len(vr))}

        try:
            import torchaudio
            a, sr = torchaudio.load(path)       # [C,S] -> [S,C]
            a = a.transpose(0,1).contiguous()
            info["audio_fps"] = int(sr)
        except Exception:
            a = None
        return v, a, info
    except Exception as e:
        raise RuntimeError(f"Failed to decode video: {path} ({e})")
    
def snippet_times(duration: float) -> List[Tuple[float,float,float]]:
 
    spans = []
    s = 0.0
    while s + TS_SEC <= duration:
        e = s + TS_SEC
        spans.append((s, e, e))  # E를 보존
        s += TS_SEC

    return spans

def sample_indices_24fps_end_aligned(e: float, file_fps: float, T: int = T_FRAMES) -> List[int]:
   
    
    times = torch.tensor([e - (i+1)/FPS_FIXED for i in range(T)][::-1], dtype=torch.float32)
    idx = (times * file_fps).round().long().tolist()
    return idx


@torch.inference_mode()
def extract_for_video_strict(
    video_path: str,
    vid_model: VideoSwinBackbone,
    aud_model_clap: AudioCLAPWindowed | None,
    img_size: int = 224,
    target_sr: int = 48_000,
    amp: bool = True,
    device: str = "cuda",
) -> Dict:
    video, audio, info =read_video_safe(video_path)  # video:(Tv,H,W,C), audio:(Sa,Ca)
    if video.numel() == 0:
        raise RuntimeError(f"decode fail: {video_path}")

    file_fps = float(info.get("video_fps", 25.0))
    total_frames = int(info.get("video_frames", video.shape[0]))
    duration = total_frames / max(1.0, file_fps)

    audio_sr = int(info.get("audio_fps", target_sr))
    wav = resample_mono(audio, orig_sr=audio_sr, target_sr=target_sr)  # (S,)

    spans = snippet_times(duration)  # (S,E,E)

    vtoks, atoks, span_list = [], [], []
    MICRO = 8 
    for i in range(0, len(spans), MICRO):
        batch_spans = spans[i:i+MICRO]

        
        v_batch = []
        for (s, e, E) in batch_spans:
            idx = sample_indices_24fps_end_aligned(E, file_fps, T_FRAMES)
            idx = [min(max(0, j), video.shape[0]-1) for j in idx]
            fr = video[idx].permute(0,3,1,2).contiguous().float() / 255.0
            fr = torch.nn.functional.interpolate(fr, size=(img_size, img_size), mode="bilinear", align_corners=False)
            v_batch.append(fr); span_list.append([s, e])
        v_batch = torch.stack(v_batch, dim=0).to(device)


        a_batch = []
        for (s, e, E) in batch_spans:
            s_win = max(0.0, E - AUDIO_WIN)
            e_win = E
            s_i, e_i = int(s_win * target_sr), int(e_win * target_sr)
            seg = wav[s_i:e_i]
            target_len = int(AUDIO_WIN * target_sr)
            if seg.numel() < target_len:
                seg = F.pad(seg, (target_len - seg.numel(), 0))  # 앞쪽 패드 (끝정렬)
            elif seg.numel() > target_len:
                seg = seg[-target_len:]
            a_batch.append(seg)
        a_batch = torch.stack(a_batch, dim=0).to(device)  # [b, N]


        with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"), dtype=torch.float16, enabled=amp):
            vtok = vid_model(v_batch)  # [b, Lv, Dv]
           
            atok = aud_model_clap(a_batch)  # 보통 [B, T', D]; 여기선 T'~=1
            if atok.dim() == 2:
                atok = atok.unsqueeze(1)

        vtoks.append(vtok.detach().to(torch.float16).cpu())
        atoks.append(atok.detach().to(torch.float16).cpu())

    vtoks = torch.cat(vtoks, dim=0) if vtoks else torch.empty(0)
    atoks = torch.cat(atoks, dim=0) if atoks else torch.empty(0)
    spans_t = torch.tensor(span_list, dtype=torch.float32) if span_list else torch.empty(0,2)

    return {
        "video_tokens": vtoks,     # [Ns, Lv, Dv]
        "audio_tokens": atoks,     # [Ns, 1(or Ta), Da]
        "spans": spans_t,          # [Ns,2] (sec)
        "meta": {
            "duration": float(duration),
            "video_fps": float(file_fps),
            "audio_sr": int(target_sr),
            "fps_fixed": FPS_FIXED,
            "t_frames": T_FRAMES,
        }
    }
from datetime import datetime
import json, csv, sys
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default='/data/jupyter/AGI/datasets/XD-violence/test/video',type=str)
    ap.add_argument("--list_path",default='/data/jupyter/AGI/encoders/multi_modal/video_audio/test.list', type=str)
    ap.add_argument("--out_dir",default='/data/jupyter/AGI/datasets/XD-violence/test/pt', type=str)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_suffix", type=str, default="")
    ap.add_argument("--videomae_ckpt", type=str, default="MCG-NJU/videomae-base")
    ap.add_argument("--clap_ckpt", type=str, default="laion/clap-htsat-fused")
    ap.add_argument("--split_name", type=str, default="unsplit")
    args = ap.parse_args()

    device = args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    out_split = ensure_dir(args.out_dir)

    # 백본 로드
    vid = VideoSwinBackbone(VideoSwinCfg(model_name=args.videomae_ckpt, frames=T_FRAMES, img_size=args.img_size, return_seq=True)).to(device)

    clap_cfg = CLAPAudioCfg(model_name=args.clap_ckpt, sr=args.sr, win_sec=0.96, hop_sec=0.96, normalize=True)
    aud_clap = AudioCLAPWindowed(clap_cfg).to(device)

    def write_report(out_dir, split_name, report):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"report_{split_name}_{ts}")
        # JSON
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # CSV (flat)
        with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fid", "status", "reason", "out_path"])
            for rec in report["records"]:
                w.writerow([rec["fid"], rec["status"], rec.get("reason",""), rec.get("out_path","")])
        print(f"[REPORT] saved -> {base}.json / {base}.csv")
  
    ids = read_list_ids(args.list_path)

    n = len(ids) if args.limit <= 0 else min(args.limit, len(ids))
    report = {
        "counts": {
            "processed": 0,
            "empty_spans": 0,
            "skipped_exists": 0,
            "not_found": 0,
            "decode_fail": 0,
            "save_fail": 0,
            "other_fail": 0,
        },
        "records": []  
    }

    pbar = tqdm(range(n), desc="Extracting", ncols=100)
    for k in pbar:
        fid = ids[k]
        status = None
        reason = ""
        out_path = os.path.join(out_split, f"{fid}{args.save_suffix}.pt")


        mp4_path = os.path.join(args.video_root, f"{fid}.mp4")
        if not os.path.exists(mp4_path):
            found = None
            for root, _, _ in os.walk(args.video_root):
                cand = os.path.join(root, f"{fid}.mp4")
                if os.path.exists(cand):
                    found = cand
                    break
            if found is None:
                status = "not_found"
                reason = f"missing: {mp4_path}"
                report["counts"]["not_found"] += 1
                report["records"].append({"fid": fid, "status": status, "reason": reason, "out_path": ""})
                pbar.set_postfix_str(status)
                continue
            mp4_path = found


        if os.path.exists(out_path):
            status = "skipped_exists"
            reason = "output already exists"
            report["counts"]["skipped_exists"] += 1
            report["records"].append({"fid": fid, "status": status, "reason": reason, "out_path": out_path})
            pbar.set_postfix_str(status)
            continue

        try:
   
            feats = extract_for_video_strict(
                video_path=mp4_path,
                vid_model=vid,
                aud_model_clap=aud_clap,
                img_size=args.img_size,
                target_sr=args.sr,
                amp=args.amp,
                device=device,
            )


            spans = feats.get("spans", None)
            is_empty = (spans is None) or (getattr(spans, "numel", lambda: 0)() == 0) or (len(spans) == 0)

    
            bin_label, multi_label = parse_label_from_id(fid)
            feats["id"] = fid
            feats["labels"] = {"binary": bin_label, "multilabel": multi_label}
            feats["meta"]["split"] = args.split_name
            feats["meta"]["src_path"] = mp4_path
            feats["meta"]["empty"] = bool(is_empty)

            
            try:
                torch.save(feats, out_path)
            except Exception as se:
                status = "save_fail"
                reason = f"torch.save failed: {se}"
                report["counts"]["save_fail"] += 1
                report["records"].append({"fid": fid, "status": status, "reason": reason, "out_path": out_path})
                pbar.set_postfix_str(status)
                continue  # 다음 파일로

          
            if is_empty:
                status = "empty_spans"
                report["counts"]["empty_spans"] += 1
            else:
                status = "processed"
                report["counts"]["processed"] += 1

            report["records"].append({"fid": fid, "status": status, "reason": reason, "out_path": out_path})
            pbar.set_postfix_str(status)

        except RuntimeError as re:
        
            status = "decode_fail"
            reason = f"{re}"
            report["counts"]["decode_fail"] += 1
            report["records"].append({"fid": fid, "status": status, "reason": reason, "out_path": ""})
            pbar.set_postfix_str(status)

        except Exception as e:
        
            status = "other_fail"
            reason = f"{e}"
            report["counts"]["other_fail"] += 1
            report["records"].append({"fid": fid, "status": status, "reason": reason, "out_path": ""})
            pbar.set_postfix_str(status)

  
    print("\n=== Extraction Summary ===")
    for k, v in report["counts"].items():
        print(f"{k:>15}: {v}")
    sys.stdout.flush()

 
    write_report(out_split, args.split_name, report)
    print("Done ->", out_split)

if __name__ == "__main__":
    main()
