import os
import subprocess
import sys

# Redirecting output to a file
orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f


# Get the list of installed packages
installed_packages = subprocess.check_output(["pip", "freeze"]).decode().splitlines()

# Iterate through the installed packages
for package in installed_packages:
    package_name = package.split('==')[0]
    try:
        # Get the installation location of the package
        package_location = subprocess.check_output(["pip", "show", package_name]).decode()
        location_line = [line for line in package_location.splitlines() if "Location" in line][0]
        package_path = location_line.split(":")[1].strip()

        # Check for any compiled shared object files (.so or .dylib) in the package folder
        for root, dirs, files in os.walk(package_path):
            for file in files:
                if file.endswith(('.so', '.dylib')):
                    file_path = os.path.join(root, file)
                    output = subprocess.check_output(["file", file_path]).decode()

                    # Check if the compiled library is for x86_64
                    if "x86_64" in output:
                        print(f"{package_name}: Compatible with x86_64")
                    else:
                        print(f"{package_name}: Incompatible architecture")

    except Exception as e:
        print(f"Error checking {package_name}: {e}")


# Restore stdout and close the file
sys.stdout = orig_stdout
f.close()
