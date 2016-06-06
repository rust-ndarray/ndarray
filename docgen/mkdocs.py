#!/usr/bin/env python3

import sys
import os
import os.path as P
import subprocess
import shutil
from subprocess import PIPE

crates = {
    "ndarray": ("docs", ""),
    "ndarray-rand": ("", "ndarray-rand"),
    #"ndarray-rblas": ("", "ndarray-rblas"),
}

def crate_name(s):
    """Return crate name (with underscores) """
    return s.replace("-", "_")

def manifest_path(crate):
    home = P.join(P.dirname(sys.argv[0]), "..")
    return P.normpath(P.join(home, crates[crate][1], "Cargo.toml"))

def run_get(cmd):
    print(cmd)
    return subprocess.Popen(cmd, stdout=PIPE).communicate()[0]

def run(cmd):
    print(cmd)
    subprocess.check_call(cmd)

def run_shell(cmd_string):
    print(cmd_string)
    subprocess.getoutput(cmd_string)

def version(crate):
    manifest = manifest_path(crate)
    pkgid = run_get(["cargo", "pkgid", "--manifest-path", manifest])
    pkgid = pkgid.decode("utf-8").strip()
    last = pkgid.rsplit("#")[-1]
    return last.rsplit(":")[-1]

def target_dir():
    home = P.join(P.dirname(sys.argv[0]), "..")
    return P.join(home, "target")

def dest_dir():
    home = P.join(P.dirname(sys.argv[0]), "..")
    return P.join(home, "master")

def doc_home():
    return P.dirname(sys.argv[0])
def image_dir():

    return P.join(doc_home(), "images")

def mkdocs():
    for crate in crates:
        run(["cargo", "doc", "-v", "--no-deps",
            "--manifest-path", manifest_path(crate),
            "--features", crates[crate][0]])
        docdir = P.join(target_dir(), "doc", crate_name(crate))
        run_shell(r'find %s -name "*.html" -exec sed -i -e "s/<title>\(.*\) - Rust/<title>%s - \1 - Rust/g" {} \;'
                  % (docdir, version(crate)))
    dest = dest_dir()
    target_doc = P.join(target_dir(), "doc")
    images_dir = image_dir()
    run_shell("rm -rvf ./%s" % dest)
    run_shell("cp -r %s %s" % (target_doc, dest))
    run_shell("cp %s/*.svg %s/ndarray/" % (images_dir, dest))
    run_shell("cat %s/custom.css >> %s/main.css" % (doc_home(), dest))
    # remove empty files
    run_shell("find %s -size 0 -delete" % (dest, ))

def main():
    path = sys.argv[0]
    os.putenv("CARGO_TARGET_DIR", target_dir())
    mkdocs()

if __name__ == "__main__":
    main()
