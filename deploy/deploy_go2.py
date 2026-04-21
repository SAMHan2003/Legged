#!/usr/bin/env python3
"""
Backward-compatible entrypoint for Go2 deployment.

This wrapper keeps the familiar `deploy_go2.py` command name, but routes all
execution through the YAML-driven deployment pipeline in `main_deploy.py`.
"""

from main_deploy import main


if __name__ == "__main__":
    main()
