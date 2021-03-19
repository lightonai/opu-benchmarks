import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opu_benchmarks",
    version="0.1.0",
    author="Giuseppe Luca Tommasone",
    author_email="luca@lighton.io",
    description="Benchmarking code for the OPU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["numpy==1.17.5", "scikit-learn==0.22.1", "scipy==1.4.1", "torch==1.2", "torchvision==0.4",
                      "pandas==0.24.2", "tqdm==4.41.1", "matplotlib==3.0.2", "Pillow==8.1.1", "lightonml==1.0.2",
                      "lightonopu==1.1.0", "networkx==2.4"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
