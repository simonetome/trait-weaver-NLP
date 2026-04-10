#!/bin/bash

mkdir -p data

if [ ! -e data/disease ]; then
    echo "Downloading disease info from OT release March 2026"
    rsync -rpltvz --delete rsync.ebi.ac.uk::pub/databases/opentargets/platform/26.03/output/disease data/.
fi


if [ ! -e data/association_by_datasource_direct ]; then
    echo "Downloading disease info from OT release March 2026"
    rsync -rpltvz --delete rsync.ebi.ac.uk::pub/databases/opentargets/platform/26.03/output/association_by_datasource_direct data/.
fi






