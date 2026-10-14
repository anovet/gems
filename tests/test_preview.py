from PIL import Image

from gems.runner import run_experiment

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_run_folder_contains_a_preview_image(synthetic):
    result = run_experiment(synthetic.config)

    preview = result.run_dir / "preview.png"
    assert preview.read_bytes()[:8] == PNG_SIGNATURE
    with Image.open(preview) as image:
        width, height = image.size
    # Main map plus a zoomed inset of fault traces: a real figure, not a thumbnail.
    assert width >= 800 and height >= 400
