from setuptools import setup, find_packages

setup(
    name='YourPackageName',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[     
        'torch==1.13.0',
        'torchvision==0.14.0',
        'tifffile==2021.7.30',
        'tqdm==4.56.0',
        'scikit-learn==1.2.2',
        'numpy==1.21.0',
        'Pillow==7.0.0',
        'scipy==1.7.2',
        'tabulate==0.8.7'
    ]
)