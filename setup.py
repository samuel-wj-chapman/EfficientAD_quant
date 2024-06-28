from setuptools import setup, find_packages

setup(
    name='YourPackageName',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[     
        'torch==2.1.0',
        'torchvision==0.16.0',
        'tifffile',
        'tqdm',
        'scikit-learn',
        'numpy==1.24.4',
        'Pillow',
        'scipy',
        'tabulate'
    ]
)