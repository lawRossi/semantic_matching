from pathlib import Path

from setuptools import setup, find_packages


if __name__ == '__main__':

    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name="semantic-matching",
        version="0.1",
        description="Semantic matching framework",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/lawRossi/semantic_matching",
        packages=find_packages(exclude=["examples", "docs", "test", "data"]),
    )
