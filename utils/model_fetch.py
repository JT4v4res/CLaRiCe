import requests
import json
import time
import threading
import logging
from typing import Optional

# Configuração de logging — ajuste o nível se quiser menos/mais verboso
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

LOG_INTERVAL_SECONDS = 1  # intervalo de atualização do contador (em segundos)


def format_elapsed(seconds: float) -> str:
    secs = int(seconds)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _timer_logger(stop_event: threading.Event, model_name: str, start_time: float, interval: int = LOG_INTERVAL_SECONDS) -> None:
    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - start_time
            logging.info(f"Downloading {model_name} — elapsed: {format_elapsed(elapsed)}")
            if stop_event.wait(interval):
                break
    finally:
        elapsed = time.monotonic() - start_time
        logging.info(f"Finished {model_name} — total elapsed: {format_elapsed(elapsed)}")


def get_models_list(session: requests.Session) -> Optional[str]:
    response = session.get("http://ollama:11434/api/tags")
    return json.loads(response.text).get("models", None)


def download_models(session: requests.Session) -> None:
    models_to_download = ["gemma3:4B", "llama3.1:8B"]

    for model in models_to_download:
        logging.info(f"Starting pull for model: {model}")
        stop_event = threading.Event()
        start_time = time.monotonic()
        timer_thread = threading.Thread(
            target=_timer_logger,
            args=(stop_event, model, start_time, LOG_INTERVAL_SECONDS),
            daemon=True,
        )
        timer_thread.start()

        try:
            response = session.post(
                "http://ollama:11434/api/pull",
                data=json.dumps({"model": model}),
            )
            logging.info(f"Response status for {model}: {response.status_code}")
            logging.info(f"Response body for {model}: {response.text}")
        except Exception as e:
            logging.exception(f"Erro ao puxar o modelo {model}: {e}")
        finally:
            stop_event.set()
            timer_thread.join(timeout=5.0)  # join seguro
            logging.info(f"Pull for {model} completed (thread joined).")


def main() -> None:
    with requests.Session() as session:
        model_list = get_models_list(session)
        if not model_list:
            download_models(session)
        else:
            logging.info("Model list já disponível; nada a baixar.")


if __name__ == "__main__":
    main()
