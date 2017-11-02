import os
from setuptools.command.install import install
from setuptools import setup
from subprocess import call

base_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(base_path, "symmetryFunctions")

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

setup(
    name="NeuralNetworks",
    version="0.1",
    description="Neural Networks Potentials",
    install_requires=["numpy", "scipy"],
    packages=["NeuralNetworks"],
    package_dir={"": "../"},
    package_data={"": ["symmetryFunctions/libSymFunSet.so"]},
    include_package_data=True,
    cmdclass={"install": CustomInstall}
)
