import subprocess
import os

showCounter = 0


def show(img, title="image"):
    global showCounter

    path = "/tmp/Imager_" + str(os.getpid()) + "_" + str(showCounter) + ".png"
    showCounter += 1

    img.save(path)

    with open(path) as imageInput:
        subprocess.Popen(["feh", "--title", title, "-"],
                         cwd="/",
                         stdin=imageInput,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    os.remove(path)
