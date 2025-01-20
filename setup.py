from setuptools import setup, find_packages

setup(
    name="llm_from_scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    author="Robin Newhouse",
    description="An LLM implementation from scratch",
    python_requires=">=3.8",
) 