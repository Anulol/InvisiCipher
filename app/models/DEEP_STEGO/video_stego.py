from pathlib import Path

import cv2
import numpy as np


def _prepare_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*.png"):
        file.unlink(missing_ok=True)


def _create_video_writer(output_video_path, fps, frame_size):
    output_video_path = Path(output_video_path)
    suffix = output_video_path.suffix.lower()

    # Prefer lossless/near-lossless codecs to preserve embedded bits.
    if suffix == ".avi":
        codec_candidates = ["FFV1", "HFYU", "MJPG", "XVID", "mp4v"]
    elif suffix == ".mp4":
        codec_candidates = ["avc1", "mp4v", "MJPG"]
    else:
        codec_candidates = ["FFV1", "HFYU", "MJPG", "XVID", "mp4v"]

    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError("Failed to create output video writer.")


def _match_frame_size(frame_bgr, target_width, target_height):
    if frame_bgr.shape[1] == target_width and frame_bgr.shape[0] == target_height:
        return frame_bgr
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _embed_secret_frame(secret_frame_bgr, cover_frame_bgr):
    # Store secret high 4 bits in cover low 4 bits for each color channel.
    secret_msb = np.right_shift(secret_frame_bgr, 4)
    cover_msb = np.bitwise_and(cover_frame_bgr, 0xF0)
    return np.bitwise_or(cover_msb, secret_msb).astype(np.uint8)


def _extract_secret_frame(stego_frame_bgr):
    # Recover secret high 4 bits and expand to full range for colored output.
    secret_nibble = np.bitwise_and(stego_frame_bgr, 0x0F)
    return np.bitwise_or(np.left_shift(secret_nibble, 4), secret_nibble).astype(np.uint8)


def hide_video(secret_video_path, cover_path, output_video_path, output_frames_dir):
    secret_video_path = Path(secret_video_path)
    cover_path = Path(cover_path)
    output_video_path = Path(output_video_path)
    output_frames_dir = Path(output_frames_dir)

    _prepare_output_dir(output_frames_dir)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    secret_cap = cv2.VideoCapture(str(secret_video_path))
    if not secret_cap.isOpened():
        raise RuntimeError("Failed to open secret video.")

    secret_width = int(secret_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    secret_height = int(secret_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = secret_cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0

    try:
        writer = _create_video_writer(output_video_path, fps, (secret_width, secret_height))
    except RuntimeError as e:
        secret_cap.release()
        raise e

    image_cover = cv2.imread(str(cover_path))
    use_image_cover = image_cover is not None
    cover_cap = None
    if not use_image_cover:
        cover_cap = cv2.VideoCapture(str(cover_path))
        if not cover_cap.isOpened():
            writer.release()
            secret_cap.release()
            raise RuntimeError("Failed to open cover video/image.")

    frame_index = 0
    while True:
        ok, secret_frame = secret_cap.read()
        if not ok:
            break

        if use_image_cover:
            cover_frame = image_cover
        else:
            ok_cover, cover_frame = cover_cap.read()
            if not ok_cover:
                cover_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_cover, cover_frame = cover_cap.read()
            if not ok_cover:
                raise RuntimeError("Cover video has no readable frames.")

        cover_frame = _match_frame_size(cover_frame, secret_width, secret_height)
        steg_frame = _embed_secret_frame(secret_frame, cover_frame)
        writer.write(steg_frame)
        cv2.imwrite(str(output_frames_dir / f"frame_{frame_index:05d}.png"), steg_frame)
        frame_index += 1

    writer.release()
    secret_cap.release()
    if cover_cap is not None:
        cover_cap.release()

    return output_video_path, output_frames_dir, frame_index


def reveal_video(stego_video_path, output_video_path, output_frames_dir):
    stego_video_path = Path(stego_video_path)
    output_video_path = Path(output_video_path)
    output_frames_dir = Path(output_frames_dir)

    _prepare_output_dir(output_frames_dir)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    stego_cap = cv2.VideoCapture(str(stego_video_path))
    if not stego_cap.isOpened():
        raise RuntimeError("Failed to open stego video.")

    width = int(stego_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stego_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stego_cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0

    try:
        writer = _create_video_writer(output_video_path, fps, (width, height))
    except RuntimeError as e:
        stego_cap.release()
        raise e

    frame_index = 0
    while True:
        ok, stego_frame = stego_cap.read()
        if not ok:
            break

        revealed_frame = _extract_secret_frame(stego_frame)
        writer.write(revealed_frame)
        cv2.imwrite(str(output_frames_dir / f"frame_{frame_index:05d}.png"), revealed_frame)
        frame_index += 1

    writer.release()
    stego_cap.release()

    return output_video_path, output_frames_dir, frame_index
