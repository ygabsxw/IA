from setuptools import setup, find_packages

setup(
    name='decision_trees',
    version='0.1.0',
    author='Gabriel Diniz Reis Vianna',
    author_email='gabrieldrvianna@gmail.com',
    description='Implementação do zero dos algoritmos ID3, C4.5 e CART.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
    python_requires='>=3.8',
)