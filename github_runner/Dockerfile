FROM ubuntu:latest

WORKDIR /opt/runner

RUN apt-get update && \
    apt-get install -y curl libdigest-sha-perl libicu-dev

COPY download-runner.sh download.sh
COPY configure-runner.sh configure.sh

RUN chmod -R 777 .
RUN useradd -m -d /opt/runner runner && echo $RUNNER_TOKEN > runner_token

USER runner

RUN ./download.sh
ENTRYPOINT cd actions-runner && ../configure.sh
