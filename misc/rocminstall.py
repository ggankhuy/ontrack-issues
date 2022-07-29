#!/usr/bin/env python3
#
# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Author: Srinivasan Subramanian (srinivasan.subramanian@amd.com)
#
# Download and install a specific ROCm version
# V1.44: Fix touch version error for 4.5.2
# V1.43: Only user packages installation ROCm 4.5 (fix dkms check)
#        Use ubuntu not xenial
# V1.42: Only user packages installation ROCm 4.5 (amdgpu separated)
# V1.41: fix ubuntu repo: change xenial to ubuntu
# V1.40: add support for 4.5 (filter out nvidia pkgs)
# V1.39: add support for 4.4.1
# V1.38: add support for 4.3.1
# V1.37: for internal repo: filter out gdb-tests
# V1.36: 4.3 release fix, do not install rccl-rdma-sharp pkg
# V1.35: Fix miopenkernels name pattern
# V1.34: Add 4.1.1 dependencies extras
# V1.33: Add support for 4.1.1
# V1.32: (for internal use) suppress rocfft-client install
# V1.31: Add --nomiopenkernels to exclude pre-built miopenkernels pkgs
# V1.30: Add --justrdc to install only RDC package
# V1.29: Add support for 4.0.1 release (uses 4.0.1 in pkg name)
# V1.28: Fix touch bug cut-and-paste typo
# V1.27: Fix touch for 3.9.1 (dot releases)
# V1.26: Add support 3.9.1 (uses 3.9.1 string in pkg name!)
# V1.25: Fix justkernel install on centos
# V1.24: Fix message - hipfot3.9.0 not installed
# V1.23: Create dummy .info/version workaround for 3.9
# V1.22: support for 3.10 release
# V1.21: rocm-dkms package not required to be installed (starting 3.9)
#        don't install openmp-extras - packaging bug in 3.9
# V1.20: rocm-dkms changed in 3.9 release
# V1.19: remove 3.8 migraphx bug fix
# V1.18: rocm 3.8 bug: migraphx package missing in debian repo
# V1.17: add --justkernel option
# V1.16: SLES repo handling
# V1.15: Fix Debian repo setup
#        Always extract rocm-dkms
# V1.14.1: Fix message
# V1.14: Allow 3.5.1 install
#        Teardown repo setup after install
# V1.13: Add baseurl option
#        Install miopenkernel packages
# V1.12: Do not install rdc ROCm package (requires gRPC preinstalled)
# V1.11: Add CentOS8/RHEL8 ROCm repo support
#        Setup repo. Don't install hipify-clang package in 3.6 - bug
#        Update usage to not dkms, kernel-headers should be preinstalled
# V1.10: Disable 3.5.1 install: 3.5.1 breaks 3.5.0 installation
#       fix nokernel flag logic on Ubuntu
# V1.9: nokernel install fixes: Always install rocm-dkms to info/version
#       Workaround for packaging bug, dependency on rocm-dkms
#       Ubuntu: dpkg-deb extract rocm-dkms in nokernel XXX HACK XXX
# V1.8: Fix for Ubuntu/Debian install to setup repo, use apt install
#       Add check for rhel OS type
#       No support for CentOS/RHEL 8
# V1.7: Add workaround for ROCm 3.5 packaging bug in CentOS
#       exclude hipify-clang package install on CentOS
# V1.6: Add --nokernel to skip rock-dkms kernel in docker installtion
#       Remove rocm-dkms along with rock-dkms*
#       Fix repourl option: set fetchurl
# V1.5: Launch install command in interactive/unbuffered mode
# V1.4: Added support for debian/ubuntu
#       SLES broken, zypper non-interactive defaults a pain
# V1.3: Use yum on SLES
#       Add support for Debian/Ubuntu
# V1.2: Install with dependencies using yum
#       Handle 3.1.1 as special case
#       Download all RPMs then install using yum (no --force install)
#       Use --skip-broken to skip broken packages
# V1.1: Initial version 4/12/2010
#   No support for Debian/Ubuntu, Only supports CentOS/SLES
#
from urllib import request
from urllib.request import urlretrieve
import re
import subprocess
import sys
import argparse
import os
import shlex
import datetime
import inspect

# Set ROCm Release Package Distribution repo baseURL below
# External ROCM release baseurl for rpm/yum: http://repo.radeon.com/rocm/yum
rocmcentos8_base = "http://repo.radeon.com/rocm/centos8/"
rocmyum_base = "http://repo.radeon.com/rocm/yum/"
rocmapt_base = "http://repo.radeon.com/rocm/apt/"
rocmzyp_base = "http://repo.radeon.com/rocm/zyp/"

# Commands
RM_F_CMD = "/bin/rm -f "
RPM_CMD = "/usr/bin/rpm"
YUM_CMD = "/usr/bin/yum"
DPKG_CMD = "/usr/bin/dpkg"
DPKGDEB_CMD = "/usr/bin/dpkg-deb"
APTGET_CMD = "/usr/bin/apt-get"
APT_CMD = "/usr/bin/apt"
APTKEY_CMD = "/usr/bin/apt-key"
WGET_CMD = "/usr/bin/wget"
ZYPPER_CMD = "/usr/bin/zypper"
ECHO_CMD = "/bin/echo"
TEE_CMD = "/usr/bin/tee"

# Package type suffixes
PKGTYPE_RPM = "rpm"
PKGTYPE_DEB = "deb"

# Supported OS types
CENTOS_TYPE = "centos" # Version 7
CENTOS8_TYPE = "centos8"
UBUNTU_TYPE = "ubuntu"
DEBIAN_TYPE = "debian"
SLES_TYPE = "sles"
RHEL_TYPE = "rhel" # Version 7

CENTOS_VERSION8_TYPESTRING = 'VERSION="8'

# OS release info
ETC_OS_RELEASE = "/etc/os-release"
# Download destination dir: default current director
DOWNLOAD_DESTDIR = "./"

#
# Is rock-dkms already installed? True if already installed
#

def print_fcn_entry(function):
    def inner(*args, **kwargs):
        fcn_args=inspect.signature(function).parameters.values()
        print("DBG: entered: ", function.__name__, "()")
        print("DBG:   parameters: ")
        for x, y in zip(args, fcn_args):
            print("DBG:    ", y,"=", x)
        function(*args, **kwargs)
        print("DBG: exited: ", function.__name__)
        return function(*args, **kwargs)
    return inner

@print_fcn_entry
def check_rock_dkms(pkgtype):
    check_cmd = {
        PKGTYPE_RPM : RPM_CMD + " -q rock-dkms ",
        PKGTYPE_DEB : DPKG_CMD + " -l rock-dkms ",
    }[pkgtype]
    if pkgtype is PKGTYPE_RPM:
        check_cmd = RPM_CMD + " -q rock-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            print(ps1.stdout.decode('utf-8'))
            return True
        except subprocess.CalledProcessError as err:
            #return False
            pass

        # check for amdgpu-dkms
        check_cmd = RPM_CMD + " -q amdgpu-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            print(ps1.stdout.decode('utf-8'))
            return True
        except subprocess.CalledProcessError as err:
            return False
    elif pkgtype is PKGTYPE_DEB:
        check_cmd = DPKG_CMD + " -l rock-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            print(ps1.stdout.decode('utf-8'))
            for line in str.splitlines(ps1.stdout.decode('utf-8')):
                if re.search(r'^i.*rock-dkms.*all', line): # 'i' for installed
                    return True
        except subprocess.CalledProcessError as err:
            return False

        check_cmd = DPKG_CMD + " -l amdgpu-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            print(ps1.stdout.decode('utf-8'))
            for line in str.splitlines(ps1.stdout.decode('utf-8')):
                if re.search(r'^i.*amdgpu-dkms.*all', line): # 'i' for installed
                    return True
            return False
        except subprocess.CalledProcessError as err:
            return False
    else:
        print("Unknown package type {}. Cannot detect rock-dkms amdgpu-dkms status.".format(pkgtype))
        return False

# Get the list of ROCm rpm packages from the ROCm release URL
pkglist = []
pkgset = set()
rockset = set()
rocklist = []
# debian/ubuntu

@print_fcn_entry
def get_deb_pkglist311(rocmurl, revstring, pkgtype):
    global pkglist
    global rocklist
    urlpath = rocmurl + "/dists/xenial/main/binary-amd64/Packages"
    try:
        urld = request.urlopen(urlpath)
        for line in str.splitlines(urld.read().decode('utf-8'), True):
            mat = re.search(rf'Filename: pool.*\.{pkgtype}', line)
            if mat:
                pkgname = line[mat.start()+len("Filename: "):mat.end()]
                # MIOpen-HIP conflicts with MIOpen-OpenCL from same repo
                if ("MIOpen-OpenCL".lower() in pkgname.lower()  # ignore, conflicts w MIOpen-HIP
                    or "MIVisionX".lower() in pkgname.lower()
                    or "hip-nvcc".lower() in pkgname.lower()
                    or "hip_nvcc".lower() in pkgname.lower()):
                        continue
                #
                if "rock-dkms".lower() in pkgname.lower():
                    rockset.add(pkgname)
                    continue
                #
                if (not re.search(rf'[a-zA-Z]+{revstring}', os.path.basename(pkgname))
                    and not re.search(rf'[a-zA-Z]+64{revstring}', os.path.basename(pkgname))):
                        pkgset.add(pkgname)
        # return set as a list
        if check_rock_dkms(pkgtype) is True:
            # remove rock-dkms and rock-dkms-firmware from list
            print(" Skipping rock-dkms: a version is already installed.")
            print((" To install rock-dkms package, please remove installed rock-dkms"
                   " first. Reboot may be required. "))
            pkglist = [ x for x in pkgset if "rock-dkms" not in x ]
#            pkglist = [ x for x in pkglist if "rocm-dkms" not in x ] # BUG WORKAROUND V1.9
            rocklist = []
        else:
            pkglist = list(pkgset)
            rocklist = list(rockset)
    except Exception as e:
        pkglist = None
        rocklist = None
        print(urlpath + " : " + str(e))

# get pkglist for 3.1.1
@print_fcn_entry
def get_pkglist311(rocmurl, revstring, pkgtype):
    global pkglist
    global rocklist
    urlpath = rocmurl
    try:
        urld = request.urlopen(urlpath)
        for line in str.splitlines(urld.read().decode('utf-8'), True):
            mat = re.search(rf'".*\.{pkgtype}"', line)
            if mat:
                pkgname = line[mat.start()+1:mat.end()-1]
                # MIOpen-HIP conflicts with MIOpen-OpenCL from same repo
                if ("MIOpen-OpenCL".lower() in pkgname.lower()
                    or "MIVisionX".lower() in pkgname.lower()
                    or "hip-nvcc".lower() in pkgname.lower()
                    or "hip_nvcc".lower() in pkgname.lower()):
                    continue
                #
                if "rock-dkms".lower() in pkgname.lower():
                    rockset.add(pkgname)
                    continue
                # TODO XXX: adjust revstring to X.Y if it is X.Y.Z
                if (not re.search(rf'[a-zA-Z]+{revstring}', pkgname)
                    and not re.search(rf'[a-zA-Z]+64{revstring}', pkgname)):
                    pkgset.add(pkgname)
        # return set as a list
        if check_rock_dkms(pkgtype) is True:
            # remove rock-dkms and rock-dkms-firmware from list
            print(" Skipping rock-dkms: a version is already installed.")
            print((" To install rock-dkms package, please remove installed rock-dkms"
                   " first. Reboot may be required. "))
            pkglist = [ x for x in pkgset if "rock-dkms" not in x ]
#            pkglist = [ x for x in pkglist if "rocm-dkms" not in x ] # BUG WORKAROUND V1.9
            rocklist = []
        else:
            pkglist = list(pkgset)
            rocklist = list(rockset)
    except Exception as e:
        pkglist = None
        rocklist = None
        print(urlpath + " : " + str(e))

#
# get pkglist for release 3.3 and newer
# select packages with names pkgX.Y.Z
#
# debian/ubuntu
@print_fcn_entry
def get_deb_pkglist(rocmurl, revstring, pkgtype):
    global pkglist
    global rocklist
    if "4.5" in revstring:
        urlpath = rocmurl + "/dists/ubuntu/main/binary-amd64/Packages"
    else:
        urlpath = rocmurl + "/dists/xenial/main/binary-amd64/Packages"
    if ("3.9.1" in revstring or "4.0.1" in revstring or "4.1.1" in revstring
        or "4.3.1" in revstring or "4.4.1" in revstring
        or "4.5.1" in revstring or "4.5.2" in revstring):
        patrevstr = revstring[0:4] # adjust search pattern to X.Y
    elif len(revstring) == 3:
        patrevstr = revstring[0:2] # adjust search pattern to X.Y
    else:
        patrevstr = revstring[0:3] # adjust search pattern to X.YY
    try:
        urld = request.urlopen(urlpath)
        for line in str.splitlines(urld.read().decode('utf-8'), True):
            mat = re.search(rf'Filename: pool.*\.{pkgtype}', line)
            if mat:
                pkgname = line[mat.start()+len("Filename: "):mat.end()]
                # MIOpen-HIP conflicts with MIOpen-OpenCL from same repo
                if ("MIOpen-OpenCL".lower() in pkgname.lower()  # ignore, conflicts w MIOpen-HIP
                    or "MIVisionX-nvcc".lower() in pkgname.lower()
                    or "hip-nvcc".lower() in pkgname.lower()
                    or "nvidia".lower() in pkgname.lower()
                    or "rdc".lower() in pkgname.lower()
                    or "rocfft-clients".lower() in pkgname.lower()
                    or "rccl-rdma-sharp".lower() in pkgname.lower()
                    or "rocm-gdb-tests".lower() in pkgname.lower()
                    or "hip_nvcc".lower() in pkgname.lower()):
                        continue
                if "rock-dkms".lower() in pkgname.lower():
                    rockset.add(pkgname)
                    continue
                #
                # Use X.Y part of X.Y.Z revstring
                # 
                if (re.search(rf'^[a-zA-Z\-]+[a-zA-Z]{patrevstr}', os.path.basename(pkgname))
                    or re.search(rf'^miopenkernel.*gfx.+db{patrevstr}', os.path.basename(pkgname))
                    or re.search(rf'^[a-zA-Z\-]+lib64{patrevstr}', os.path.basename(pkgname))):
                        pkgset.add(pkgname)
                        continue
                # Starting 3.9 release, only one rocm-dkms to go with rock-dkms
                # and rocm-dkms is optional package
        # return set as a list
        if check_rock_dkms(pkgtype) is True:
            # remove rock-dkms and rock-dkms-firmware from list
            print(" Skipping rock-dkms: a version is already installed.")
            print((" To install rock-dkms package, please remove installed rock-dkms"
                   " first. Reboot may be required. "))
            pkglist = [ x for x in pkgset if "rock-dkms" not in x ]
#            pkglist = [ x for x in pkglist if "rocm-dkms" not in x ] # BUG WORKAROUND V1.9
            rocklist = []
        else:
            pkglist = list(pkgset)
            rocklist = list(rockset)
            pkgn = [ x for x in pkglist if "rocm-dkms" in x ]
            if pkgn: # remove rocm-dkms from rocklist
                rocklist = [x for x in rocklist if "rocm-dkms" not in x ]
    except Exception as e:
        pkglist = None
        rocklist = None
        print(urlpath + " : " + str(e))

# justrdc
@print_fcn_entry
def get_deb_justrdc_pkglist(rocmurl, revstring, pkgtype):
    global pkglist
    global rocklist
    if "4.5" in revstring:
        urlpath = rocmurl + "/dists/ubuntu/main/binary-amd64/Packages"
    else:
        urlpath = rocmurl + "/dists/xenial/main/binary-amd64/Packages"
    if ("3.9.1" in revstring or "4.0.1" in revstring or "4.1.1" in revstring
        or "4.3.1" in revstring or "4.4.1" in revstring
        or "4.5.1" in revstring or "4.5.2" in revstring):
        patrevstr = revstring[0:4] # adjust search pattern to X.Y
    elif len(revstring) == 3:
        patrevstr = revstring[0:2] # adjust search pattern to X.Y
    else:
        patrevstr = revstring[0:3] # adjust search pattern to X.YY
    try:
        urld = request.urlopen(urlpath)
        for line in str.splitlines(urld.read().decode('utf-8'), True):
            mat = re.search(rf'Filename: pool.*\.{pkgtype}', line)
            if mat:
                pkgname = line[mat.start()+len("Filename: "):mat.end()]
                # MIOpen-HIP conflicts with MIOpen-OpenCL from same repo
                if ("rdc".lower() not in pkgname.lower()):
                    continue

                #
                # Use X.Y part of X.Y.Z revstring
                # 
                if (re.search(rf'^[a-zA-Z\-]+[a-zA-Z]{patrevstr}', os.path.basename(pkgname))
                    or re.search(rf'^miopenkernel.*gfx.+db{patrevstr}', os.path.basename(pkgname))
                    or re.search(rf'^[a-zA-Z\-]+lib64{patrevstr}', os.path.basename(pkgname))):
                        pkgset.add(pkgname)
                        continue
                # Starting 3.9 release, only one rocm-dkms to go with rock-dkms
                # and rocm-dkms is optional package
        # return set as a list
        pkglist = list(pkgset)
        rocklist = list(rockset)
    except Exception as e:
        pkglist = None
        rocklist = None
        print(urlpath + " : " + str(e))

@print_fcn_entry
def get_justrdc_pkglist(rocmurl, revstring, pkgtype):
    global pkglist
    global rocklist
    urlpath = rocmurl
    if ("3.9.1" in revstring or "4.0.1" in revstring or "4.1.1" in revstring
        or "4.3.1" in revstring or "4.4.1" in revstring
        or "4.5.1" in revstring or "4.5.2" in revstring):
        patrevstr = revstring[0:4] # adjust search pattern to X.Y.Z
    elif len(revstring) == 3:
        patrevstr = revstring[0:2] # adjust pat to X.Y
    else:
        patrevstr = revstring[0:3] # adjust pat to X.YY
    try:
        urld = request.urlopen(urlpath)
        for line in str.splitlines(urld.read().decode('utf-8'), True):
            mat = re.search(rf'".*\.{pkgtype}"', line)
            if mat:
                pkgname = line[mat.start()+1:mat.end()-1]
                # MIOpen-HIP conflicts with MIOpen-OpenCL from same repo
                # default is MIOpen-HIP
                if ("rdc".lower() not in pkgname.lower()):
                        continue
                #
                # Use X.Y part of X.Y.Z revstring
                # 
                if (re.search(rf'^[a-zA-Z\-]+[a-zA-Z]{patrevstr}', pkgname)
                    or re.search(rf'^miopenkernel.*gfx.+db{patrevstr}', pkgname)
                    or re.search(rf'^[a-zA-Z\-]+lib64{patrevstr}', pkgname)):
                        pkgset.add(pkgname)
                # Starting 3.9 release, only one rocm-dkms to go with rock-dkms
                # and rocm-dkms is optional package
        # return set as a list
        pkglist = list(pkgset)
        rocklist = list(rockset)
    except Exception as e:
        pkglist = None
        rocklist = None
        print(urlpath + " : " + str(e))

@print_fcn_entry
def get_pkglist(rocmurl, revstring, pkgtype):
    global pkglist
    global rocklist
    urlpath = rocmurl
    if ("3.9.1" in revstring or "4.0.1" in revstring or "4.1.1" in revstring
        or "4.3.1" in revstring or "4.4.1" in revstring
        or "4.5.1" in revstring or "4.5.2" in revstring):
        patrevstr = revstring[0:4] # adjust search pattern to X.Y.Z
    elif len(revstring) == 3:
        patrevstr = revstring[0:2] # adjust pat to X.Y
    else:
        patrevstr = revstring[0:3] # adjust pat to X.YY
    try:
        urld = request.urlopen(urlpath)
        for line in str.splitlines(urld.read().decode('utf-8'), True):
            mat = re.search(rf'".*\.{pkgtype}"', line)
            if mat:
                pkgname = line[mat.start()+1:mat.end()-1]
                # MIOpen-HIP conflicts with MIOpen-OpenCL from same repo
                # default is MIOpen-HIP
                if ("MIOpen-OpenCL".lower() in pkgname.lower()
                    or "MIVisionX".lower() in pkgname.lower()
                    or "hip-nvcc".lower() in pkgname.lower()
                    or "nvidia".lower() in pkgname.lower()
                    or "rdc".lower() in pkgname.lower()
                    or "rocfft-clients".lower() in pkgname.lower()
                    or "rocm-gdb-tests".lower() in pkgname.lower()
                    or "rccl-rdma-sharp".lower() in pkgname.lower()
                    or "hip_nvcc".lower() in pkgname.lower()):
                        continue
                if "rock-dkms".lower() in pkgname.lower():
                    rockset.add(pkgname)
                    continue
                #
                # Use X.Y part of X.Y.Z revstring
                # 
                if (re.search(rf'^[a-zA-Z\-]+[a-zA-Z]{patrevstr}', pkgname)
                    or re.search(rf'^miopenkernel.*gfx.+db{patrevstr}', pkgname)
                    or re.search(rf'^[a-zA-Z\-]+lib64{patrevstr}', pkgname)):
                        pkgset.add(pkgname)
                # Starting 3.9 release, only one rocm-dkms to go with rock-dkms
                # and rocm-dkms is optional package
        # return set as a list
        if check_rock_dkms(pkgtype) is True:
            # remove rock-dkms and rock-dkms-firmware from list
            print(" Skipping rock-dkms: a version is already installed.")
            print((" To install rock-dkms package, please remove installed rock-dkms"
                   " first. Reboot may be required. "))
            pkglist = [ x for x in pkgset if "rock-dkms" not in x ]
#            pkglist = [ x for x in pkglist if "rocm-dkms" not in x ] # BUG WORKAROUND V1.9
            rocklist = []
        else:
            pkglist = list(pkgset)
            rocklist = list(rockset)
            pkgn = [ x for x in pkglist if "rocm-dkms" in x ]
            if pkgn: # remove rocm-dkms from rocklist
                rocklist = [x for x in rocklist if "rocm-dkms" not in x ]
    except Exception as e:
        pkglist = None
        rocklist = None
        print(urlpath + " : " + str(e))

# Workaround: create .info/version dummy file
@print_fcn_entry
def workaround_dummy_versionfile_deb(args, rocmbaseurl):
    global pkglist
    global rocklist

    if args.justkernel is True:
        return

    if (args.revstring[0] == "3.9.1" or args.revstring[0] == "4.0.1" or args.revstring[0] == "4.1.1"
        or args.revstring[0] == "4.3.1" or args.revstring[0] == "4.4.1"
        or args.revstring[0] == "4.5.1" or args.revstring[0] == "4.5.2"):
        touchcmd = "touch /opt/rocm-" + args.revstring[0] + "/.info/version"
    else:
        touchcmd = "touch /opt/rocm-" + args.revstring[0] + ".0/.info/version"
    try:
        ps1 = subprocess.Popen(touchcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)

@print_fcn_entry
def workaround_dummy_versionfile_rpm(args, rocmbaseurl):
    global pkglist
    global rocklist

    if args.justkernel is True:
        return

    if (args.revstring[0] == "3.9.1" or args.revstring[0] == "4.0.1" or args.revstring[0] == "4.1.1"
        or args.revstring[0] == "4.3.1" or args.revstring[0] == "4.4.1"
        or args.revstring[0] == "4.5.1" or args.revstring[0] == "4.5.2"):
        touchcmd = "touch /opt/rocm-" + args.revstring[0] + "/.info/version"
    else:
        touchcmd = "touch /opt/rocm-" + args.revstring[0] + ".0/.info/version"
    try:
        ps1 = subprocess.Popen(touchcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)


# Download and install packages utility functions
@print_fcn_entry
def download_and_install_nodeps_rpm(args, rocmbaseurl, pkgname):
    global pkglist
    global rocklist
    if args.repourl:
        fetchurl = args.repourl[0] + "/"
    else:
        if args.baseurl is None:
            fetchurl = rocmbaseurl + "/" + args.revstring[0] + "/"
        else:
            fetchurl = rocmbaseurl + "/"
    rmcmd = RM_F_CMD
    rpmicmd = RPM_CMD + " -ivh --nodeps "
    # download destdir
    urlretrieve(fetchurl + pkgname, args.destdir[0] + "/" + os.path.basename(pkgname))
    execcmd = rpmicmd + args.destdir[0] + "/" + os.path.basename(pkgname) + " "
    rmcmd = rmcmd + args.destdir[0] + "/" + os.path.basename(pkgname) + " "
    try:
        ps1 = subprocess.Popen(execcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
    try:
        ps2 = subprocess.Popen(rmcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        print(" Unexpected error encountered! Did you forget sudo?")

# HACK: Ubuntu doesn't like broken packages. So extract rocm-dkms
@print_fcn_entry
def download_and_extract_nodeps_deb(args, rocmbaseurl, pkgname):
    global pkglist
    global rocklist
    if args.repourl:
        fetchurl = args.repourl[0] + "/"
    else:
        if args.baseurl is None:
            fetchurl = rocmbaseurl + "/" + args.revstring[0] + "/"
        else:
            fetchurl = rocmbaseurl + "/"
    rmcmd = RM_F_CMD
    dpkggetcmd = DPKGDEB_CMD + " -xv "
    # download destdir
    urlretrieve(fetchurl + pkgname, args.destdir[0] + "/" + os.path.basename(pkgname))
    execcmd = dpkggetcmd + args.destdir[0] + "/" + os.path.basename(pkgname) + " "
    execcmd = execcmd + " / "  # extract under root dir /
    rmcmd = rmcmd + args.destdir[0] + "/" + os.path.basename(pkgname) + " "
    try:
        ps1 = subprocess.Popen(execcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
    try:
        ps2 = subprocess.Popen(rmcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        print(" Unexpected error encountered! Did you forget sudo?")


@print_fcn_entry
def download_and_install_deb(args, rocmbaseurl, pkgname):
    global pkglist
    global rocklist
    if args.repourl:
        fetchurl = args.repourl[0] + "/"
    else:
        if args.baseurl is None:
            fetchurl = rocmbaseurl + "/" + args.revstring[0] + "/"
        else:
            fetchurl = rocmbaseurl + "/"
    rmcmd = RM_F_CMD
    aptinstcmd = APT_CMD + " install -y "
    aptgetcmd = APTGET_CMD + " -y -f install "
    # download destdir
    urlretrieve(fetchurl + pkgname, args.destdir[0] + "/" + os.path.basename(pkgname))
    execcmd = aptinstcmd + args.destdir[0] + "/" + os.path.basename(pkgname) + " "
    rmcmd = rmcmd + args.destdir[0] + "/" + os.path.basename(pkgname) + " "
    try:
        ps1 = subprocess.Popen(execcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
    try:
        ps1 = subprocess.Popen(aptgetcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
    try:
        ps2 = subprocess.Popen(rmcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        print(" Unexpected error encountered! Did you forget sudo?")

@print_fcn_entry
def setup_sles_zypp_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # Set up rocm repo for chosen rev to install
        # use rev specific rocm repo
        zypprepo = "[rocm" + args.revstring[0] + "]\nenabled=1\nautorefresh=0\nbaseurl=" + fetchurl + "\ntype=rpm-md\ngpgcheck=0"
        echocmd = ECHO_CMD + " -e '" + zypprepo + "' "
        repofilename = "/etc/zypp/repos.d/rocm" + args.revstring[0] + ".repo "
        teecmd = TEE_CMD + " " + repofilename + " "
        try:
            ps1 = subprocess.Popen(shlex.split(echocmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            ps2 = subprocess.Popen(teecmd.split(), stdin=ps1.stdout, stdout=subprocess.PIPE)
            ps1.stdout.close()
            out = ps2.communicate()[0]
            print(out.decode('utf-8'))
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)
        # zypper install ROCm dependencies - starting with ROCm 4.1.1!
        zypprefresh = ZYPPER_CMD + " addrepo https://download.opensuse.org/repositories/devel:languages:perl/SLE_15/devel:languages:perl.repo"
        try:
            ps1 = subprocess.Popen(zypprefresh.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        # apt update repo
        zypprefresh = ZYPPER_CMD + " refresh "
        try:
            ps1 = subprocess.Popen(zypprefresh.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

@print_fcn_entry
def setup_centos_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # Set up rocm repo for chosen rev to install
        # use rev specific rocm repo
        yumrepo = "[ROCm" + args.revstring[0] +"]\nname=ROCm\nbaseurl=" + fetchurl + "\nenabled=1\ngpgcheck=0"
        echocmd = ECHO_CMD + " -e '" + yumrepo + "' "
        repofilename = "/etc/yum.repos.d/rocm" + args.revstring[0] + ".repo"
        teecmd = TEE_CMD + " " + repofilename + " "
        try:
            ps1 = subprocess.Popen(shlex.split(echocmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            ps2 = subprocess.Popen(teecmd.split(), stdin=ps1.stdout, stdout=subprocess.PIPE)
            ps1.stdout.close()
            out = ps2.communicate()[0]
            print(out.decode('utf-8'))
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        # install ROCm dependencies - starting with ROCm 4.1.1!!
        yumupdate = YUM_CMD + " install --assumeyes perl-File-Which perl-File-BaseDir perl-File-Copy-Recursive perl-URI-Encode "
        try:
            ps1 = subprocess.Popen(yumupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        # clean repo
        yumupdate = YUM_CMD + " clean all "
        try:
            ps1 = subprocess.Popen(yumupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

@print_fcn_entry
def setup_centos8_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # install ROCm dependencies - enable powertools - starting with ROCm 4.1.1!!
        yumupdate = YUM_CMD + " config-manager --set-enabled powertools"
        try:
            ps1 = subprocess.Popen(yumupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        setup_centos_repo(args, fetchurl)

def remove_centos_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # remove rocm repo for chosen rev to install
        # use rev specific rocm repo
        repofilename = "/etc/yum.repos.d/rocm" + args.revstring[0] + ".repo"
        rmrepocmd = RM_F_CMD + " " + repofilename
        try:
            ps1 = subprocess.Popen(rmrepocmd.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)
        # clean repo
        yumupdate = YUM_CMD + " clean all "
        try:
            ps1 = subprocess.Popen(yumupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

def remove_sles_zypp_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # remove rocm repo for chosen rev to install
        # use rev specific rocm repo
        repofilename = "/etc/zypp/repos.d/rocm" + args.revstring[0] + ".repo "
        rmrepocmd = RM_F_CMD + " " + repofilename
        try:
            ps1 = subprocess.Popen(rmrepocmd.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)
        # clean repo
        zyppclean = ZYPPER_CMD + " clean "
        try:
            ps1 = subprocess.Popen(zyppclean.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

def setup_debian_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
    else:
        # Set up rocm repo for chosen rev to install
        wgetkey = WGET_CMD + " -q -O - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key "
        aptkeycmd = APTKEY_CMD + " add -"
        try:
            ps1 = subprocess.Popen(wgetkey.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            ps2 = subprocess.Popen(aptkeycmd.split(), stdin=ps1.stdout, stdout=subprocess.PIPE)
            ps1.stdout.close()
            out = ps2.communicate()[0]
            print(out.decode('utf-8'))
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        # use rev specific rocm repo
        if "4.5" in args.revstring[0]:
            debrepo = "deb [arch=amd64] " + fetchurl + " ubuntu main "
        else:
            debrepo = "deb [arch=amd64] " + fetchurl + " xenial main "
        echocmd = ECHO_CMD + " '" + debrepo + "' "
        repofilename = "/etc/apt/sources.list.d/rocm" + args.revstring[0] + ".list "
        teecmd = TEE_CMD + " " + repofilename
        try:
            ps1 = subprocess.Popen(shlex.split(echocmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            ps2 = subprocess.Popen(teecmd.split(), stdin=ps1.stdout, stdout=subprocess.PIPE)
            ps1.stdout.close()
            out = ps2.communicate()[0]
            print(out.decode('utf-8'))
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)
        # apt update repo
        aptupdate = APT_CMD + " clean"
        try:
            ps1 = subprocess.Popen(aptupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        # apt update repo
        aptupdate = APT_CMD + " update"
        try:
            ps1 = subprocess.Popen(aptupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)

        # install ROCm dependencies - starting with ROCm 4.1.1!!
        aptupdate = APTGET_CMD + " -y install libfile-which-perl libfile-basedir-perl libfile-copy-recursive-perl liburi-encode-perl"
        try:
            ps1 = subprocess.Popen(aptupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)



def remove_debian_repo(args, fetchurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # remove rocm repo for chosen rev to install
        # use rev specific rocm repo
        repofilename = "/etc/apt/sources.list.d/rocm" + args.revstring[0] + ".list "
        rmrepocmd = RM_F_CMD + " " + repofilename
        try:
            ps1 = subprocess.Popen(rmrepocmd.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)
        # clean repo
        # apt update repo
        aptupdate = APT_CMD + " clean"
        try:
            ps1 = subprocess.Popen(aptupdate.split(), bufsize=0).communicate()[0]
        except subprocess.CalledProcessError as err:
            for line in str.splitlines(err.output.decode('utf-8')):
                print(line)


# On Ubuntu, use the steps below to install downloaded deb
# 1. Install rock-dkms-firmware apt install -y rock-dkms-firmware
# 2. Install rock-dkms: apt install -y rock-dkms
# 3. Fix rock-dkms install dependencies: apt-get -y -f install
# 4. Remove rock-dkms-firmware, rock-dkms
# 5. Install the remaining downloaded deb: apt install -y *.deb
# 6. Remove remaining deb packages
def download_install_rocm_deb(args, rocmbaseurl):
    global pkglist
    global rocklist

    if args.repourl:
        pass
        # use rev specific rocm repo
    else:
        # Force nodeps install of  rocm-dkms package even if --nokernel is True
        # if args.nokernel is True:
        # Always extract rocm-dkms, don't install 
        pkgn = [ x for x in pkglist if "rocm-dkms" in x ]
        if pkgn:
            download_and_extract_nodeps_deb(args, rocmbaseurl, pkgn[0])
        pkglist = [ x for x in pkglist if "rocm-dkms" not in x ]

        if args.baseurl is None:
            fetchurl = rocmbaseurl + "/" + args.revstring[0] + "/"
        else:
            fetchurl = rocmbaseurl + "/"
        # Set up rocm repo for chosen rev to install
        setup_debian_repo(args, fetchurl)

    # Download and install from custom repo URL
    rmcmd = RM_F_CMD
    aptinstcmd = APT_CMD + " install -y "
    aptgetcmd = APTGET_CMD + " -y -f install "
    # Force nodeps install of  rocm-dkms package even if --nokernel is True
    if args.nokernel is False:
        # install rock-dkms-firmware first
        pkgn = [ x for x in rocklist if "rock-dkms-firmware" in x ]
        if pkgn:
            # remove rock-dkms-firmware from list
            rocklist = [ x for x in rocklist if "rock-dkms-firmware" not in x ]
            # Download and Install rock-dkms-firmware first (assumes only one)
            download_and_install_deb(args, rocmbaseurl, pkgn[0])
        pkgn = [ x for x in rocklist if "rock-dkms" in x ]
        if pkgn:
            rocklist = [ x for x in rocklist if "rock-dkms" not in x ]
            # Download and Install rock-dkms
            download_and_install_deb(args, rocmbaseurl, pkgn[0])
        pkgn = [ x for x in rocklist if "rocm-dkms" in x ]
        if pkgn:
            download_and_install_deb(args, rocmbaseurl, pkgn[0])
    else:
        pkglist = [ x for x in pkglist if "rocm-dkms" not in x ]

    if args.justkernel is True:
        return

    # Install the rest of the deb packages
    if args.repourl:
        fetchurl = args.repourl[0] + "/"
    else:
        if args.baseurl is None:
            fetchurl = rocmbaseurl + "/" + args.revstring[0] + "/"
        else:
            fetchurl = rocmbaseurl + "/"
    rmcmd = RM_F_CMD
    aptinstcmd = APT_CMD + " install -y "
    execcmd = aptinstcmd
    # rearrange pkglist to put rocm-dkms rocm-dev first
    rocmdkms = [ x for x in pkglist if "rocm-dkms" in x ]
    pkglist = [ x for x in pkglist if "rocm-dkms" not in x ]
    rocmdev = [ x for x in pkglist if "rocm-dev" in x ]
    pkglist = [ x for x in pkglist if "rocm-dev" not in x ]
    pkglist = rocmdkms + rocmdev + pkglist
    for n in pkglist:
        # download destdir
        urlretrieve(fetchurl + n, args.destdir[0] + "/" + os.path.basename(n))
        execcmd = execcmd + args.destdir[0] + "/" + os.path.basename(n) + " "
        rmcmd = rmcmd + args.destdir[0] + "/" + os.path.basename(n) + " "
        print('.', end='', flush=True)

    try:
        ps1 = subprocess.Popen(execcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)

    try:
        ps1 = subprocess.Popen(aptinstcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)

    try:
        ps2 = subprocess.Popen(rmcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        print(" Unexpected error encountered! Did you forget sudo?")

#
# --rev REV is the ROCm version number string
# --destdir DESTDIR directory to download rpm for installation
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('[V1.44]rocminstall.py: utility to '
        ' download and install ROCm packages for specified rev'
        ' (dkms, kernel headers must be installed, requires sudo privilege) '),
        prefix_chars='-')
    parser.add_argument('--rev', nargs=1, dest='revstring', default='rpm',
        help=('specifies ROCm release repo to use '
              ' as in http://repo.radeon.com/rocm/{apt, yum, zyp, centos8}/<REV> '
              ' Example: --rev 3.5 for ROCm 3.5 repo '
              ' http://repo.radeon.com/rocm/{apt, yum, zyp, centos8}/3.5, '
              ' or '
              ' --rev 3.3 for ROCm 3.3 repo '
              ' http://repo.radeon.com/rocm/{apt, yum, zyp}/3.3 ')
              )
    parser.add_argument('--destdir', nargs=1, dest='destdir', default='.',
        help=('specify directory where to download RPM'
              ' before installation. Default: current directory '
              ' --destdir /tmp to use /tmp directory')
              )
    parser.add_argument('--list', dest='listonly', action='store_true',
        help=('just list the packages that will be installed'
              ' -- do not download or install ')
              )
    parser.add_argument('--repourl', nargs=1, dest='repourl', default=None,
        help=('specify ROCm repo URL to use from where to download packages'
              ' Example: --repourl http://compute-artifactory/build/xyz')
              )
    parser.add_argument('--baseurl', nargs=1, dest='baseurl', default=None,
        help=('specify early access ROCm repo URL to use from where to download packages'
              ' Example: --baseurl http://repo.radeon.com/rocm/private/apt_3.6-priv/')
              )
    parser.add_argument('--nokernel', dest='nokernel', action='store_true',
        help=('do not install rock kernel packages, for example, '
              ' used to install ROCm in docker')
              )
    parser.add_argument('--justkernel', dest='justkernel', action='store_true',
        help=('ONLY install rock kernel packages of specified version '
              ' - undefined behavior if --nokernel also specified')
              )
    parser.add_argument('--justrdc', dest='justrdc', action='store_true',
        help=('ONLY install ROCm Radeon Data Center Monitor tool      '
              ' - attempts to install rdcX.Y.Z package corresponding to rev')
              )
    parser.add_argument('--nomiopenkernels', dest='nomiopenkernels', action='store_true',
        help=('do not install pre-built miopenkernels packages        '
              ' - saves space and installation time')
              )

    args = parser.parse_args();

    # BUG: ROCm 3.5.1 release breaks 3.5.0 installation!
    if "3.5.1" in args.revstring:
        print("WARNING: ROCm 3.5.1 breaks 3.5.0 installations!")

    # BUG: ROCm 3.9.1 release breaks 3.9.0 installation!
    if "3.9.1" in args.revstring:
        print("WARNING: ROCm 3.9.1 breaks 3.9.0 installations!")

    if "4.5" in args.revstring:
        print("NOTE: Starting with ROCM 4.5 kernel/amdgpu dkms not installed!")
        args.nokernel = True

    # Determine installed OS type
    ostype = None
    with open(ETC_OS_RELEASE, 'r') as f:
        for line in f:
            if CENTOS_TYPE.lower() in line.lower():
                ostype = CENTOS_TYPE
                break
            if DEBIAN_TYPE.lower() in line.lower():
                ostype = UBUNTU_TYPE
                break
            if UBUNTU_TYPE.lower() in line.lower():
                ostype = UBUNTU_TYPE
                break
            if SLES_TYPE.lower() in line.lower():
                ostype = SLES_TYPE
                break
            if RHEL_TYPE.lower() in line.lower():
                ostype = CENTOS_TYPE
                break

    # Detect CentOS8/RHEL8 to set the correct ROCm repo
    if ostype is CENTOS_TYPE:
        with open(ETC_OS_RELEASE, 'r') as f:
            for line in f:
                if CENTOS_VERSION8_TYPESTRING.lower() in line.lower():
                    ostype = CENTOS8_TYPE
                    break

    if ostype is None:
        print("Exiting: Unknown installed OS type")
        parser.print_help()
        sys.exit(1)

    # Log version and date of run
    print("Running V1.44 rocminstall.py utility for OS: " + ostype + " on: " + str(datetime.datetime.now()))

    #
    # Set pkgtype to use based on ostype
    #
    pkgtype = None
    pkgtype = {
        CENTOS8_TYPE : PKGTYPE_RPM,
        CENTOS_TYPE : PKGTYPE_RPM,
        UBUNTU_TYPE : PKGTYPE_DEB,
        SLES_TYPE : PKGTYPE_RPM
    }[ostype]

    #
    # Get the list of package names from repo using --rev option
    #
    if args.repourl:
        rocmbaseurl = args.repourl[0]
    else:
        if args.baseurl:
            rocmbaseurl = args.baseurl[0]
        else:
            rocmbaseurl = {
                CENTOS8_TYPE : rocmcentos8_base,
                CENTOS_TYPE : rocmyum_base,
                UBUNTU_TYPE : rocmapt_base,
                SLES_TYPE : rocmzyp_base
            }[ostype]

    if "3.1." in args.revstring[0]:
        if pkgtype is PKGTYPE_DEB:
            get_deb_pkglist311(rocmbaseurl + "/" + args.revstring[0], args.revstring[0],
                pkgtype)
        else:
            get_pkglist311(rocmbaseurl + "/" + args.revstring[0], args.revstring[0],
                pkgtype)
    elif args.repourl:
        get_pkglist(args.repourl[0] + "/", args.revstring[0], pkgtype)
    elif args.baseurl:
        if pkgtype is PKGTYPE_DEB:
            get_deb_pkglist(rocmbaseurl, args.revstring[0], pkgtype)
        else:
            get_pkglist(rocmbaseurl, args.revstring[0], pkgtype)
    else:
        if pkgtype is PKGTYPE_DEB:
            if args.justrdc is True:
                get_deb_justrdc_pkglist(rocmbaseurl + "/" + args.revstring[0], args.revstring[0], pkgtype)
            else:
                get_deb_pkglist(rocmbaseurl + "/" + args.revstring[0], args.revstring[0], pkgtype)
        else:
            if args.justrdc is True:
                get_justrdc_pkglist(rocmbaseurl + "/" + args.revstring[0], args.revstring[0], pkgtype)
            else:
                get_pkglist(rocmbaseurl + "/" + args.revstring[0], args.revstring[0], pkgtype)

    # V1.7: XXX Workaround for ROCm 3.5 on CentOS, V1.11 Workaround 3.6
    if ((ostype is CENTOS_TYPE  or ostype is RHEL_TYPE or ostype is CENTOS8_TYPE)
        and ("3.5" in args.revstring[0] or "3.6" in args.revstring[0])):
        # exclude hipify-clang
        pkglist = [ x for x in pkglist if "hipify-clang" not in x ]
        print("NOTE: Not installing hipify-clang RPM due to packaging issue.")
        print("NOTE: Please install hipify-clang RPM manually using: ")
        print("NOTE:  sudo rpm -ivh --force hipify-clang3.5.0-11.0.0.x86_64.rpm ")

    # V1.21 XXX Workaround for ROCm 3.9 Packaging hipfort, openmp-extras packaging bug XXX
    if ("3.9" in args.revstring[0]):
        # exclude openmp-extras package due to packaging BUG
        pkglist = [ x for x in pkglist if "openmp-extra" not in x ]
        pkglist = [ x for x in pkglist if "hipfort" not in x ]
        print("NOTE: Not installing hipfort3.9.0 package due to ROCm 3.9 bug.")
        print("NOTE: Please install hipfot3.9.0 manually using force option")

    #
    # Based on os type, set the package install command and options
    #
    cmd = {
       CENTOS8_TYPE : YUM_CMD + " localinstall --skip-broken --assumeyes ",
       CENTOS_TYPE : YUM_CMD + " localinstall --skip-broken --assumeyes ",
       UBUNTU_TYPE : APTGET_CMD + " --no-download --ignore-missing -y install ",
       SLES_TYPE : ZYPPER_CMD + " --no-gpg-checks install --allow-unsigned-rpm --no-recommends "
    }[ostype]

    #
    # If pkglist is None, print help and exit
    #
    if pkglist is None:
        parser.print_help()
        sys.exit(1)

    # if no miopenkernels option, remove miopenkernels packages
    if args.nomiopenkernels is True:
        pkglist = [ x for x in pkglist if "miopenkernel" not in x ]

    #
    # If --list specified, print the package list and exit
    #
    if args.listonly is True:
        print("List of packages selected:\n")
        if args.nokernel is True:
            pass
#            pkglist = [ x for x in pkglist if "rocm-dkms" not in x ] # BUG WORKAROUND V1.9
        else:
            print('\n'.join(sorted(rocklist)) + '\n')
        print('\n'.join(sorted(pkglist)))
        sys.exit(0)

    #
    # Download and install the selected packages
    # Ubuntu/Debian dpkg/apt does not have an easy method to use dpkg, apt
    # to install downloaded deb packages with dependencies equivalent
    # yum localinstall command on CentOS, SLES.
    # So call a debian specific function for Ubuntu/Debian
    #
    # for Ubuntu/Debian
    if ostype is UBUNTU_TYPE:
        download_install_rocm_deb(args, rocmbaseurl)
        workaround_dummy_versionfile_deb(args, rocmbaseurl)
        remove_debian_repo(args, rocmbaseurl)
        sys.exit(0)

    # for CentOS and SLES use the following
    execcmd = cmd
    rmcmd = RM_F_CMD
    if args.repourl:
        fetchurl = args.repourl[0] + "/"
    else:
        if args.baseurl is None:
            fetchurl = rocmbaseurl + "/" + args.revstring[0] + "/"
        else:
            fetchurl = rocmbaseurl + "/"

    if ostype is CENTOS_TYPE:
        setup_centos_repo(args, fetchurl)
    elif ostype is CENTOS8_TYPE:
        setup_centos8_repo(args, fetchurl)
    else:
        setup_sles_zypp_repo(args, fetchurl)
    # skip if --nokernel option is True
    if args.nokernel is True:
        # For install rocm-dkms as workaround for bug in packaging
        pkgn = [ x for x in pkglist if "rocm-dkms" in x ]
        if pkgn:
            download_and_install_nodeps_rpm(args, rocmbaseurl, pkgn[0])
        pkglist = [ x for x in pkglist if "rocm-dkms" not in x ]
    else:
        for n in sorted(rocklist):
            # download to destdir
            urlretrieve(fetchurl + n, args.destdir[0] + "/" + os.path.basename(n))
            execcmd = execcmd + args.destdir[0] + "/" + os.path.basename(n) + " "
            rmcmd = rmcmd + args.destdir[0] + "/" + os.path.basename(n) + " "
            print('.', end='', flush=True)

    if args.justkernel is False:
        for n in sorted(pkglist):
            # download to destdir
            urlretrieve(fetchurl + n, args.destdir[0] + "/" + os.path.basename(n))
            execcmd = execcmd + args.destdir[0] + "/" + os.path.basename(n) + " "
            rmcmd = rmcmd + args.destdir[0] + "/" + os.path.basename(n) + " "
            print('.', end='', flush=True)

    try:
        ps1 = subprocess.Popen(execcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        print(" Unexpected error encountered! Did you forget sudo?")

    try:
        ps2 = subprocess.Popen(rmcmd.split(), bufsize=0).communicate()[0]
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
#
    if ostype is CENTOS_TYPE or ostype is CENTOS8_TYPE:
        remove_centos_repo(args, fetchurl)
    else:
        remove_sles_zypp_repo(args, fetchurl)
    workaround_dummy_versionfile_rpm(args, fetchurl)

    sys.exit(0)
