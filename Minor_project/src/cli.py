"""
Local command-line interface (uses the persistent library).

    python -m src.cli add path/to/video.mp4 --id lecture1 [--no-visual]
    python -m src.cli list
    python -m src.cli search "what is backpropagation" [--mode dense|bm25|hybrid] [--modality speech|visual|both]
    python -m src.cli ask "explain the diagram on the CNN slide"
    python -m src.cli delete lecture1
"""
import argparse

from .library import Library, sanitize_video_id


def main():
    ap = argparse.ArgumentParser(prog="insightrag")
    ap.add_argument("--library", default=None, help="library folder (default ./library or $INSIGHTRAG_LIBRARY)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("video")
    a.add_argument("--id", default=None)
    a.add_argument("--no-visual", action="store_true")
    a.add_argument("--interval", type=float, default=20.0)
    a.add_argument("--whisper", default="base")

    sub.add_parser("list")

    d = sub.add_parser("delete")
    d.add_argument("video_id")

    for name in ("search", "ask"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--video", default=None)
        s.add_argument("-k", type=int, default=5)
        s.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
        s.add_argument("--modality", default="both", choices=["speech", "visual", "both"])

    args = ap.parse_args()
    lib = Library(args.library)

    if args.cmd == "add":
        import os
        vid = sanitize_video_id(args.id or os.path.splitext(os.path.basename(args.video))[0])
        info = lib.add_video(args.video, vid, use_visual=not args.no_visual,
                             frame_interval=args.interval, whisper_model=args.whisper,
                             progress=lambda f, t: print(f"[{int(f * 100):3d}%] {t}"))
        print(f"Indexed '{vid}': {info['speech_chunks']} speech + {info['visual_chunks']} visual chunks")
        for w in info["warnings"]:
            print("warning:", w)
    elif args.cmd == "list":
        for v, s in lib.load_index().stats().items():
            print(v, s)
    elif args.cmd == "delete":
        lib.delete_video(args.video_id)
        print("deleted", args.video_id)
    else:
        from .answer import generate_answer
        from .cache import DiskCache
        from .chunking import format_timestamp

        mods = ("speech", "visual") if args.modality == "both" else (args.modality,)
        res = lib.load_index().search(args.query, top_k=args.k, mode=args.mode,
                                      modalities=mods, video_id=args.video)
        if args.cmd == "ask":
            try:
                ans = generate_answer(args.query, res, cache=DiskCache(str(lib.cache_dir)))
            except Exception as e:  # noqa: BLE001
                ans = None
                print(f"Could not write an answer: {e}\nShowing the closest moments instead.\n")
            if ans:
                v = ans["validation"]
                print(ans["text"], f"\n\n[citations valid: {v['n_valid']}/{v['n_citations']}]\n")
                if ans["abstained"]:
                    return
        for r in res:
            print(f"{r['video_id']:>12} {format_timestamp(r['start']):>8} {r['modality']:<6} "
                  f"rel={r['rel']:.2f}  {r['text'][:110]}")


if __name__ == "__main__":
    main()
