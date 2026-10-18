"""Module entry point; workers live in an importable module for Windows spawn."""
from validation.study import main

if __name__ == "__main__":
    main()
