"""Check the configured live visual Qwen API without starting simulation."""
import argparse
import json

from PIL import Image, ImageDraw

from .qwen import QwenClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    args = parser.parse_args()
    client = QwenClient(args.config)
    image = Image.new('RGB', (96, 96), 'white')
    ImageDraw.Draw(image).rectangle((20, 20, 76, 76), fill='blue')
    result = client.request('api_check', 'Return JSON only.',
                            'What color is the square? Return {"square_color":"blue|red|other"}.', [image])
    if result.get('square_color') != 'blue':
        raise RuntimeError('API responded but failed the simple image capability check.')
    print(json.dumps(dict(passed=True, model=client.model, base_url=client.base_url,
                          image_understanding=result), ensure_ascii=False))


if __name__ == '__main__':
    main()
