from setuptools import setup, find_packages
import logging
logger = logging.getLogger(__name__)

name = 'bloch_sphere'
package_name = 'bloch_sphere'
version = '0.4.0'

try:
    with open('README.md', 'r') as f:
        long_desc = f.read()
except:
    logger.warning('Could not open README.md.  long_description will be set to None.')
    long_desc = None

setup(
    name = package_name,
    packages = find_packages(),
    entry_points = {
        'console_scripts': [
            'animate_bloch=bloch_sphere.animate_bloch:run_from_command_line',
            'animate_bloch2=bloch_sphere.animate_bloch_compare:run_from_command_line',
        ]},
    version = version,
    description = 'Visualization tools for the qubit Bloch sphere',
    long_description = long_desc,
    long_description_content_type = 'text/markdown',
    author = 'Casey Duckering',
    #author_email = '',
    url = f'https://github.com/cduck/{name}',
    download_url = f'https://github.com/cduck/{name}/archive/{version}.tar.gz',
    keywords = ['quantum computing', 'qubit', 'jupyter'],
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Framework :: IPython',
        'Framework :: Jupyter',
    ],
    install_requires = [
        'numpy',
        'drawSvg',
        'hyperbolic',
        'latextools',
    ],
)

