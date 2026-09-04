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

The generated value is an example deployment secret. Never commit a real
pseudonym key to the repository, place it in configuration examples, or expose
it in logs.

## Key storage

Store pseudonym keys in a deployment-specific secret-management system or
another access-controlled secret store.

Deployments should:

- restrict access to the minimum services and operators that require the key;
- keep keys separate from source code, datasets, reports, and backups where
  practical;
- never log pseudonym keys;
- never commit keys to Git;
- avoid copying keys into tickets, documentation, or diagnostic output.

## Key rotation

Rotate the pseudonym key according to the deployment's security policy and
immediately when compromise is suspected.

Rotation changes the HMAC output. The same trip identifier processed with a new
key therefore produces a different pseudonym.

This intentionally limits long-term linkability, but it also means records
created under different keys cannot be joined by pseudonym alone.

Old keys should be retired after the approved transition period and deleted
when they are no longer required by the deployment's retention policy.

## Compromise response

If a pseudonym key may have been exposed:

1. stop using the affected key;
2. generate and deploy a new key;
3. identify affected data, logs, backups, and reports;
4. revoke access to the compromised secret where applicable;
5. retire and delete the old key when retention requirements permit;
6. review whether affected pseudonymized records require deletion or regeneration.

## Trace-data retention

Deployments must define explicit retention periods.

- **Raw GPX:** retain only for the minimum period needed for authorized processing.
- **Prepared traces and `PreparedTrace.points`:** treat as sensitive location data.
- **Reports and exports:** apply an explicit retention period.
- **Thresholded aggregates:** may be retained longer only after review confirms
  they contain no raw traces or directly identifying information.

Backups and temporary files must follow compatible deletion schedules.

## Logging requirements

Logs must never contain:

- raw device identifiers;
- pseudonym keys or other secrets;
- raw private trace coordinates;
- unnecessary trip identifiers.

## Deployment checklist

Before processing private traces, confirm that:

- [ ] consent and privacy checks occur before trace processing;
- [ ] only trip-scoped identifiers enter OpenTrace;
- [ ] no raw device identifiers are stored or logged;
- [ ] the pseudonym key is stored outside source control and logs;
- [ ] a key-rotation schedule is documented;
- [ ] compromise-response procedures are documented;
- [ ] old-key retirement and deletion behavior are documented;
- [ ] raw GPX retention and deletion periods are defined;
- [ ] prepared-trace retention and deletion periods are defined;
- [ ] report/export retention periods are defined;
- [ ] backup and temporary-file deletion behavior is defined;
- [ ] only reviewed, thresholded aggregate outputs are published.