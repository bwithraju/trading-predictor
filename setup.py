from setuptools import setup, find_packages

setup(
    name='trading_predictor',
    version='0.2',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': ['trading-predictor=main:app'],
    }
)