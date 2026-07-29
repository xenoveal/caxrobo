"""
Reviewer plug-ins (v0.3.0 Phase 5). Empty on purpose — see
plugins/__init__.py's docstring: framework.registry.load_all() walks this
package with pkgutil and imports each leaf module itself, so this __init__
must NOT eagerly import trade_quality (a broken reviewer would otherwise break
every other plug-in import too). Its only job is making `reviewers` a real
subpackage so pkgutil.walk_packages finds it.
"""
