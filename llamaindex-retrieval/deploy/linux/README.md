# Linux deployment

The Linux deployment keeps application code and runtime data separate:

- application: `/workspace/rwkvrag`
- FineWiki parquet files: `/workspace/finewiki`
- OpenSearch, MongoDB, uploads and logs: `/workspace/rwkvrag-data`

The supplied Supervisor configuration runs MongoDB, OpenSearch and the API without systemd.
OpenSearch only listens on loopback; the management API listens on port `8090`.

Useful commands:

```bash
cd /workspace/rwkvrag/llamaindex-retrieval
.venv/bin/python -m pip install --no-deps --force-reinstall .
sudo supervisorctl -c /workspace/rwkvrag/llamaindex-retrieval/deploy/linux/supervisord.conf status
sudo supervisorctl -c /workspace/rwkvrag/llamaindex-retrieval/deploy/linux/supervisord.conf restart rwkvrag
tail -f /workspace/rwkvrag-data/logs/rwkvrag.log
```

Run the local package installation command after every source update. The production virtual
environment uses an installed wheel, so copying files under `src/` alone does not update the API.
