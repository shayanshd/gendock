#!/bin/sh
bash service redis-server start
screen -dmS "Celery" bash worker-entrypoint.sh
bash server-entrypoint.sh
