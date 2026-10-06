# The dashboard over a project's results, read-only.
#
#   docker build -t tripwire-dashboard .
#   docker run --rm -p 8501:8501 -v "$PWD:/project" tripwire-dashboard
#
# The project (tripwire.toml, datasets, prompts and tripwire.db) is mounted, not copied:
# the image holds only the tool. Another config file: append `--config experiments/zoo.toml`.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"

WORKDIR /project
EXPOSE 8501
ENTRYPOINT ["tripwire", "dashboard", "--read-only", "--port", "8501"]
