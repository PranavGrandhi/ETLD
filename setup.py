import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="etld",
    version="1.0.0",
    author="He Wang",
    author_email="w1047181605@stu.xjtu.edu.cn",
    description="encoder-transformation layer-decoder model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xjtu-xsbsy/ETLD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=["matplotlib", "numba", "numpy", "scipy", "torch", "pandas"],
)