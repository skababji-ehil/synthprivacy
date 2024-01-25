from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()



setup(
    name="synthprivacy",
    version="1.0.0",
    description="A calculator of membership disclosure privacy risk associated with synthetic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EHIL",
    author_email="skababji@ehealthinformation.ca",
    classifiers=[
        "Development Status :: 4 - Beta", "Programming Language :: Python :: 3.10","Operating System :: OS Independent","License :: OSI Approved :: MIT License"
    ],
    keywords="synthetic, clinical trials, generative, risk, privacy, membership disclosure",
    project_urls = {
        'Home': 'https://github.com/skababji-ehil/fuzzy_sql'
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "jupyter",
        "multiprocess"
    ],
    extras_require={
        "dev": ["wheel","dvc", "sphinx","sphinxcontrib-bibtex","sphinxcontrib-napoleon"],
    },

)
