#!/bin/bash

npm run build:all
cp .htaccess build/.
rsync -av --delete-after build/ pyvrciz@ssh.cluster028.hosting.ovh.net:/home/pyvrciz/nexus
