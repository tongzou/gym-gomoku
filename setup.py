from setuptools import setup, find_packages

setup(name='gym_gomoku',
      version='0.4',
      url='https://github.com/tongzou/gym-gomoku.git',
      author='Tong Zou',
      license='MIT',
      packages=find_packages(),
      install_requires=['gym', 'numpy', 'six']
)
