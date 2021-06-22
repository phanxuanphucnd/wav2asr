import os
from setuptools import setup, find_packages

from arizona_asr.version import __version__

try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements

long_description = ''

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]

VERSION = os.getenv('PACKAGE_VERSION', __version__)[1:]

setup(
    name='arizona_asr',
    version=VERSION,
    description='Arizona-ASR is a toolbox for Automatic speech recognition.',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='',
    packages=find_packages(),
    include_package_data=True,
    author='Phanxuan Phuc',
    author_email='phanxuanphucnd@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.x',
    ],
    install_requires=reqs,
    keywords='arizona_asr',
    python_requires='>=3.6',
    py_modules=['arizona_asr'],
    entry_points={
        'console_scripts': [
            'arizona_asr = arizona_asr.run_cli:entry_point'
        ]
    },
)