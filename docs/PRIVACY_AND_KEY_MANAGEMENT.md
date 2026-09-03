# Privacy, pseudonym key rotation, and trace-data retention

OpenTrace ML treats private GPS traces and prepared trace points as sensitive
data. Pseudonymization reduces direct identification risk, but it does not make
trace data anonymous.

## Threat model

Deployments should assume that an attacker may obtain:

- application logs;
- exported reports;
- prepared trace records;
- backups or temporary files;
- a pseudonym key through accidental disclosure or host compromise.

A pseudonym must not be treated as proof that a person or trip cannot be
re-identified. Location history can remain identifying even when direct
identifiers have been removed.

OpenTrace deployments therefore must:

- keep raw device identifiers outside OpenTrace;
- use trip-scoped identifiers instead of persistent hardware identifiers;
- keep pseudonym keys outside source control and logs;
- restrict access to raw GPX and `PreparedTrace.points`;
- publish only reviewed, thresholded aggregate outputs.

## Pseudonymization is not anonymization

OpenTrace uses HMAC-SHA256 to replace a raw trip identifier with a keyed
pseudonym.

This protects the original identifier from direct disclosure, but the resulting
trace can still contain sensitive location patterns. Repeated locations, home
or work areas, timestamps, and route history may enable re-identification.

Pseudonymized trace data must therefore continue to be handled as sensitive
data.

## Key generation

Generate pseudonym keys using a cryptographically secure random source.

Example using Python:

```bash
python -c "import secrets; print(secrets.token_hex(32))"