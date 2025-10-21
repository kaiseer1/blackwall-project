"""
Setup script for BlackWall v4.0
"""

from setuptools import setup, find_packages

with open("README_NEW.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="blackwall",
    version="4.0.0",
    author="Basil Abdullah, Enhanced by Claude AI",
    author_email="444019967@stu.bu.edu.sa",
    description="AI-Driven Cybersecurity Defense System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/blackwall",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "imbalanced-learn>=0.9.0",
        "scapy>=2.4.5",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "psutil>=5.8.0",
        "colorama>=0.4.4",
        "rich>=10.0.0",
        "PyYAML>=6.0",
        "requests>=2.26.0",
        "python-dateutil>=2.8.0",
    ],
    entry_points={
        "console_scripts": [
            "blackwall=blackwall_new:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "web/*.html"],
    },
)
