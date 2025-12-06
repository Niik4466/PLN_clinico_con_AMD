from setuptools import setup, find_packages

setup(
    name="GPUMonitor",
    version="0.1.0",
    packages=find_packages(),
    description="A shared GPU monitoring library for AMD and NVIDIA",
    author="Niik4466",
    install_requires=[
        # Dependencies are handled in the Dockerfiles/requirements.txt
        # 'amdsmi', # AMD only
        # 'pynvml', # NVIDIA only
    ],
)
