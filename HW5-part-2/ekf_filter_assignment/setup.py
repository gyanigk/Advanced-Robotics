from setuptools import find_packages, setup

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="filtering_exercises",
    version="0.1.0",
    author="Brendan",
    author_email="your.email@example.com",
    description="Filtering exercises for robotics, split into three assignments: Bayes Filter, Extended Kalman Filter, and Particle Filter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/filtering_exercises",
    packages=find_packages(),
    package_data={
        "filtering_exercises": [
            "assignment1_bayes/*.py",
            "assignment1_bayes/tests/*.py",
            "assignment1_bayes/writeup/*.tex",
            "assignment2_ekf/*.py",
            "assignment2_ekf/tests/*.py",
            "assignment2_ekf/writeup/*.tex",
            "assignment3_particle/*.py",
            "assignment3_particle/tests/*.py",
            "assignment3_particle/writeup/*.tex",
            "environments/*.py",
            "utils/*.py",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "assignment1": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "assignment2": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "scipy>=1.7.0",  # For linear algebra operations
        ],
        "assignment3": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "scipy>=1.7.0",  # For statistical functions
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
