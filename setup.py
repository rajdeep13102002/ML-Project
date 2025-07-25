from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'

def get_requirements(file_path:str)->List[str]: # Corrected type hint for List
    '''
    This Function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # FIX: Changed requirements.txt to requirements
        requirements = [req.replace("\n","") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements        

setup(
    name = 'ML Project',
    version = '0.0.1',
    author = 'Rajdeep',
    author_email = 'umranirajdeep13@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)