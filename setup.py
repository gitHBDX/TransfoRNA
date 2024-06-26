from setuptools import find_packages, setup

setup(
    name='TransfoRNA',
    version='0.0.1',
    description='TransfoRNA: Navigating the Uncertainties of Small RNA Annotation with an Adaptive Machine Learning Strategy',
    url='https://github.com/gitHBDX/TransfoRNA',
    author='YasserTaha,JuliaJehn',
    author_email='ytaha@hb-dx.com,jjehn@hb-dx.com,tsikosek@hb-dx.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Biological Researchers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(include=['transforna', 'transforna.*']),
    install_requires=[
        "anndata==0.8.0",
        "dill==0.3.6",
        "hydra-core==1.3.0",
        "imbalanced-learn==0.9.1",
        "matplotlib==3.5.3",
        "numpy==1.22.3",
        "omegaconf==2.2.2",
        "pandas==1.5.2",
        "plotly==5.10.0",
        "PyYAML==6.0",
        "rich==12.6.0",
        "viennarna>=2.5.0a5",
        "scanpy==1.9.1",
        "scikit_learn==1.2.0",
        "torch==1.10.1",
        "skorch==0.12.1",
        "tensorboard==2.16.2",
        "Levenshtein==0.21.0",
        "huggingface_hub==0.23.3",
        "transformers==4.41.2",
        "biopython==1.78",
    ],
    #add dependency link for torch
    dependency_links=[
        "https://download.pytorch.org/whl/cu113",
    ],
    python_requires='>=3.9',
    #move yaml files to package
    package_data={'': ['*.yaml']},
)
