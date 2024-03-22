from setuptools import setup


def read_requirements():
    with open('requirements.txt', 'r') as req:
        content = req.read()
        requirements = content.split('\n')

    return requirements


setup(
    name='npp',
    version='1.0',
    packages=['pre_processing'],
    url='',
    license='',
    author='Antônio Pereira',
    author_email='antonio258p@gmail.com',
    description='Python natural language pre processing',
    install_requires=read_requirements()
)