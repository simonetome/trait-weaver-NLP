#!/bin/bash

mkdir -p data
mkdir -p data/efo_embeddings

if [ ! -e data/disease ]; then
    echo "Downloading disease info from OT release June 2026"
    rsync -rpltvz --delete rsync.ebi.ac.uk::pub/databases/opentargets/platform/26.06/output/disease data/.
fi


#if [ ! -e data/association_by_datasource_indirect ]; then
#    echo "Downloading disease info from OT release June 2026"
#    rsync -rpltvz --delete rsync.ebi.ac.uk::pub/databases/opentargets/platform/26.06/output/association_by_datasource_indirect data/.
#fi


#if [ ! -e data/efo_embeddings/embeddings* ]; then
wget https://github.com/simonetome/efo-embeddings/releases/download/Embeddings/embeddings_256.zip -P data/efo_embeddings
unzip -o data/efo_embeddings/embeddings_256.zip -d data/efo_embeddings
rm data/efo_embeddings/embeddings_256.zip
#fi





