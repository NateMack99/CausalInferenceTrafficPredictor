#!/bin/bash

wget https://data.cityofchicago.org/api/v3/views/gc7y-n4xa/query.json
mv query.json daily_traffic.json
wget https://data.cityofchicago.org/api/v3/views/85ca-t3if/query.json
mv query.json crashes.json
