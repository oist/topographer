# Security Policy

## Maintenance Expectations & Supported Versions

This package is a community-driven scientific tool. It is maintained on a best-effort basis as a research project, not a commercial product. We do not have a dedicated security team or guaranteed response windows.

We only support and patch the **latest minor release** available on PyPI. 

| Version | Supported | Notes |
| ------- | --------- | ----- |
| Latest  | :white_check_mark: Yes | Patches are only applied to the most recent release. |
| Older   | :x: No | Please upgrade to the latest version to resolve issues. |

## Definition of a Vulnerability (Scope)

Given the nature of scientific computing, we distinguish between standard software bugs and true security vulnerabilities.

### :inbox_tray: What is IN SCOPE:
* Remote Code Execution (RCE) flaws triggered *without* loading untrusted local files.
* Severe data corruption flaws that silently alter scientific computation outputs without throwing errors.

### :no_entry: What is OUT OF SCOPE:
* **Untrusted Data Input:** If a user manually downloads a malicious data file (e.g., NumPy binary, Pickle file, HDF5, CSV) from an untrusted source and passes it to this package, any resulting system compromise is considered an environment-level issue. Users must verify their own input data sources.
* **Denial of Service (DoS):** Script crashes, unhandled exceptions, high memory usage, or infinite loops caused by extreme scientific datasets are treated as standard performance bugs, not security threats.

## Reporting a Vulnerability

**Please do not open a public GitHub Issue for potential security flaws.**

Because this project is maintained in our spare time, a public disclosure gives us no time to fix the issue before it can be exploited. Instead, please report flaws through one of these private channels:

1. **GitHub Private Vulnerability Reporting:** Navigate to the "Security" tab of this repository, click "Advisories", and select "Report a vulnerability". (Highly Preferred)
2. **Direct Contact:** Open a private dialogue with the maintainer named in the package metadata (`Igors Dubanevics`, https://orcid.org/0000-0002-9989-6649) through the repository issues URL only after the report is already private, or use GitHub Private Vulnerability Reporting.

### Please include:
* A minimal, reproducible script utilizing standard Python libraries.
* The specific versions of this package, Python, and key dependencies (e.g., NumPy, SciPy) being used.

## Our Process

* **Acknowledgment:** We will review your report when our schedule allows. There is no automated or immediate response system.
* **Fixes:** If verified, a fix will be bundled directly into the next standard tag/release of the package. 
* **Advisories:** We will publish a GitHub Security Advisory alongside the new release to credit your discovery.
