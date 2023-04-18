from setuptools import setup # type: ignore

setup(
    name="gym_rad_search",
    version="0.1",
    install_requires=[
        "gym",
        "matplotlib",
        "numpy",
        "typing-extensions",
        "visilibity @ git+https://github.com/peproctor/PyVisiLibity.git@c76020079110231f882f38f61b3ab25d01de21f0",
    ],
)
