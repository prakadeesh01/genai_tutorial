from setuptools import find_packages, setup

setup(
    name="Ecommercebot",
    version="0.0.1",
    author="pragi",
    author_email="prakadeesh01@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain','langchain-community',
    'langchain-huggingface','datasets','pypdf','python-dotenv','flask','langchain-google-genai',
    'google-generativeai']
)