

This is VoxBench.



### Volume & Fidelity Process

Run the following scripts:

```python
python change_volume.py -j JSON_LOG_PATH -r ROOT_DIR
python reduce_fidelity.py -j JSON_LOG_PATH -r ROOT_DIR
```

`JSON_LOG_PATH`: The json process log file, i.e. corresponding `processed_dialog.json`.

`ROOT_DIR`: The root directory of generated audio files
