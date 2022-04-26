FROM windpioneers/gdal-python:familiar-catshark-gdal-2.4.1-python-3.9-dev

# Tell zsh where you want to store history
#     We leave you to decide, but if you put this into a folder that's been mapped
#     into the container, then history will persist over container rebuilds :)
#
#     !!!IMPORTANT!!!
#     Make sure your .zsh_history file is NOT committed into your repository, as it can contain
#     sensitive information. So in this case, you should add
#         .devcontainer/.zsh_history
#     to your .gitignore file.
#
ENV HISTFILE="/workspaces/aerosense-tools/.devcontainer/.zsh_history"

# Switch to vscode user
USER vscode

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH "/home/vscode/.poetry/bin:$PATH"
RUN poetry config virtualenvs.create false

# Install python dependencies
COPY pyproject.toml poetry.lock ./
# RUN poetry install

# Ensure pre-commit is activated
# RUN pre-commit install && pre-commit install -t commit-msg
