import os
from setuptools.command.install import install
from setuptools.command.build_py import build_py
from setuptools import setup
from subprocess import call

base_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(base_path, "NeuralNetworks/descriptors")

class CustomInstall(install):
    """
    CustomInstall class following the suggestion in:
    https://stackoverflow.com/questions/1754966/how-can-i-run-a-makefile-in-setup-py
    """
    def run(self):

        def compile_library():
            call("make", cwd=lib_path)
        self.execute(compile_library, [],  "Compiling shared library")
        # copy all the files to the /usr/local/lib/python2.7/dist-package
        install.run(self)

class CustomBuild(build_py):
    """
    CustomInstall build based of the example in CustomInstall
    """
    def run(self):

        def compile_library():
            call("make", cwd=lib_path)
        self.execute(compile_library, [],  "Compiling shared library")
        build_py.run(self)

setup(
    name="NeuralNetworks",
    version="0.1",
    description="Neural Networks Potentials",
    setup_requires=["numpy"],
    install_requires=["numpy", "scipy", "progressbar2", "tensorflow", "matplotlib"],
    packages=[
        "NeuralNetworks",
        "NeuralNetworks.data_generation",
        "NeuralNetworks.optimize",
        "NeuralNetworks.md_utils",
        "NeuralNetworks.md_utils.ode",
        "NeuralNetworks.md_utils.pset",
        "NeuralNetworks.descriptors",
        "NeuralNetworks.types"
    ],
    package_dir={
        "NeuralNetworks": "NeuralNetworks", 
        "NeuralNetworks.data_generation": "NeuralNetworks/data_generation", 
        "NeuralNetworks.optimize": "NeuralNetworks/optimize",
        "NeuralNetworks.md_utils": "NeuralNetworks/md_utils",
        "NeuralNetworks.md_utils.ode": "NeuralNetworks/md_utils/ode",
        "NeuralNetworks.md_utils.pset": "NeuralNetworks/md_utils/pset",
        "NeuralNetworks.descriptors": "NeuralNetworks/descriptors",
        "NeuralNetworks.types": "NeuralNetworks/types"
    },
    package_data={"NeuralNetworks.descriptors": ["libSymFunSet.so"]},
    include_package_data=True,
    cmdclass={"install": CustomInstall, "build_py": CustomBuild}
)
