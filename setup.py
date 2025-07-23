import setuptools

version = '0.0.1'

if __name__ == '__main__':
    setuptools.setup(
        name='openandn',
        version=version,
        description='Open Artificial Neural Detector Networks',
        long_description='',
        author='Anatoli Starostin',
        author_email='anatoli.starostin@gmail.com',
        package_dir={"": "src"},
        packages=[
            "openandn",
            "openandn.util"
        ],
    )
