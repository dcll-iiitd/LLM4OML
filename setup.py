from setuptools import setup, find_packages

setup(
    name="convergence-proof-agent",
    version="3.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain-nvidia-ai-endpoints>=0.1.0",
        "langchain-core>=0.1.0",
        "langgraph>=0.1.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    python_requires=">=3.9",
)