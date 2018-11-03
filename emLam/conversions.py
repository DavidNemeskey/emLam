#!/usr/bin/env python3
"""Conversion functions from tsv to the final token format."""

from __future__ import absolute_import, division, print_function
import inspect
import json
from operator import itemgetter
import re
import sys

from emLam import WORD, LEMMA, POS, ANAS


_pos_str = r'(\[[^[+]+\])'
_pos_re = re.compile(_pos_str)
_pos_pos_re = re.compile(r'/([^]]+)\]|(Adj\|nat)\]')
_pos_hyph_re = re.compile(r'^\[Hyph:[^]]+\]|\[Punct\]$')
# To fix erroneous tagging
_pos_map = {
    '[N]': '[/N]',
    '[V]': '[/V]',
    '[Num]': '[/Num]',
    '[_Mod]': '[_Mod/V]',
}
# Drop the 'default' inflections (zero morphemes) + degenerate tags
_pos_drop = {'[Nom]', '[Prs.NDef.3Sg]', '[]'}

# _anas_eq = re.compile(r'(?P<deep>.+?)(?P<tag>\[[^]]+\])(?:=(?P<surface>.+?))?')
_anas_eq = re.compile(r'(?:([^+]+){0}(?:=([^+]+))?)|(?:(\+){0}=(\+))|(?:(){0}())'.format(_pos_str))


def field_word(fields):
    return [fields[WORD]]


def field_lemma(fields):
    return [fields[LEMMA]]


def field_full_pos(fields):
    """The full POS field."""
    return [fields[POS]]


def field_pos_inf(fields):
    """The POS tag + the inflections as separate tokens."""
    pos = fields[POS]
    if '[' in pos and not _pos_hyph_re.match(pos):  # not OTHER, maybe sth else too
        pos_ana = [None]
        infl_ana = []
        form = 0
        parts = _pos_re.findall(pos)
        for part in parts:
            part = _pos_map.get(part, part)
            if part == '[/Supl]':  # Delete 'leg', but remember it
                form = 2
            elif part.startswith('[_Comp'):
                form = max(form, 1)
            posm = _pos_pos_re.search(part)
            if posm:
                pos_ana[0] = posm.group(1) or posm.group(2)
            else:
                if part not in _pos_drop:
                    infl_ana.append(part)
        if form == 1:
            pos_ana.append('[Comp]')
        elif form == 2:
            pos_ana.append('[Supl]')
        return pos_ana + infl_ana
    else:
        return ['OTHER']


def field_lemma_deriv_kr(fields):
    """ lemma_deriv k r"""
    kr = fields[2]
    parts = kr.rsplit('/')
    to_add = []
    if len(parts) > 1:
        for p in range(len(parts[:-1]) - 1, -1, -1):
            if '[COMPAR]' in parts[p]:
                to_add.append('<COMPAR>')
                del parts[p]
            elif '[COMPAR_DESIGN]' in parts[p]:
                to_add.append('<COMPAR>')
                to_add.append('<DESIGN>')
                del parts[p]
            elif '[SUPERLAT]' in parts[p]:
                to_add.append('<SUPERLAT>')
                del parts[p]
            elif '[SUPERLAT_DESIGN]' in parts[p]:
                to_add.append('<SUPERLAT>')
                to_add.append('<DESIGN>')
                del parts[p]
            elif '[ORD]' in parts[p]:
                to_add.append('<ORD>')
                del parts[p]

    if len(parts) > 1:
        ret = [fields[1] + '_' + '/'.join(parts[:-1])]
    else:
        ret = [fields[1]]

    for false_deriv in to_add:
        ret.append(false_deriv)

    last_index = parts[-1].find('<')
    if last_index > 0:
        while True:
            index = parts[-1].find('><', last_index)
            if index > 0:
                ret.append(parts[-1][last_index:index + 1])
                last_index = index + 1
            else:
                ret.append(parts[-1][last_index:])
                break
    return ret


def field_pos_kr(fields):
    kr = fields[2]
    last_index = kr.find('<')
    if last_index > 0:
        ret = [kr[:last_index]]
        while True:
            index = kr.find('><', last_index)
            if index > 0:
                ret.append(kr[last_index:index + 1])
                last_index = index + 1
            else:
                ret.append(kr[last_index:])
                break
        return ret
    else:
        return [kr]


def field_lemma_inf(fields):
    """
    For POS tags returned by GATE: ~lemmad_krs.
    Fields: word, lemma, pos (, ...).
    """
    return _field_lemma_inf(fields)


def field_lemma_inf_with_pos(fields):
    """
    For POS tags returned by GATE: ~lemmad_krs, but the POS tag
    (in the narrow sense, i.e. /N, etc.) is appended to the lemma as well.
    Fields: word, lemma, pos (, ...).
    """
    return _field_lemma_inf(fields, True)


def _field_lemma_inf(fields, keep_pos=False):
    lemma = fields[LEMMA]
    pos = fields[POS]
    word_ana = [lemma]
    infl_ana = []
    if '[' in pos and not _pos_hyph_re.match(pos):  # not OTHER, maybe sth else too
        form = 0
        parts = _pos_re.findall(pos)
        for part in parts:
            part = _pos_map.get(part, part)
            if part == '[/Supl]':  # Delete 'leg', but remember it
                form = 2
            if part.startswith('[/'):  # POS tag
                if keep_pos:
                    word_ana[-1] += part
            elif part.startswith('[_Comp'):
                form = max(form, 1)
            elif part.startswith('[_'):
                word_ana[-1] += part
            else:
                if part not in _pos_drop:
                    infl_ana.append(part)
        if form == 1:
            word_ana.append('[Comp]')
        elif form == 2:
            word_ana.append('[Supl]')
    return word_ana + infl_ana


def field_real_lemma_inf(fields):
    """
    For POS tags returned by GATE. The main difference between this and
    lemma_inf is that the derivational suffixes are not exposed; instead, their
    "surface" forms are fused with the lemma. In other words, this function
    always returns valid words as lemmas, while lemma_inf does not.
    """
    word, lemma, pos, anas = fields[WORD], fields[LEMMA], fields[POS], fields[ANAS]
    anas = json.loads(anas)
    # not OTHER, hyphen, or unanalyzed word
    if '[' in pos and not _pos_hyph_re.match(pos) and anas:
        # Filter to the analyses selected by emLemma
        anas = [ana for ana in anas if ana['lemma'] == lemma and ana['feats'] == pos]
        for ana in anas:
            parts = _split_ana(ana)
            ret_lemma, ret_infl = _reconstruct_lemma_inf(parts)
            return [ret_lemma] + ret_infl
    else:
        return [word]


def _split_ana(ana):
    """
    Returns the list of tag -- deep form -- surface form tuples in the analysis.
    """
    ret = []
    for part in _anas_eq.finditer(ana['ana']):
        groups = part.groups()
        if groups[1]:
            deep, tag = groups[0:2]
            surface = groups[2] if groups[2] else groups[0]
        elif groups[4]:
            deep, tag, surface = groups[3:6]
        elif groups[7]:
            deep, tag, surface = groups[6:9]
        else:
            assert 'This should not happen: {}'.format(part.groups())

        ret.append((tag, deep, surface))
    return ret


def _reconstruct_lemma_inf(ana_parts):
    """
    Reconstructs the lemma and inflection based on the ana
    parts (POS tag -> surface form mapping) we extracted from the analysis.

    Note that the proper handling of comparative and superlative forms of
    needs more work, and is not handled by the current code.
    """
    lemma = []
    inf = []

    # For the last non-inflection tag, we must keep the deep ("lemmatized")
    # form, so that the word doesn't end in a linking/modified vowel
    keep_surface = False
    for ana_tag, ana_deep, ana_surface in ana_parts[::-1]:
        # Note: 
        if ana_tag.startswith('[/') or ana_tag.startswith('[_'):
            # POS tag or derivation: fuse with lemma
            lemma.append(ana_surface if keep_surface else ana_deep)
            keep_surface = True
        elif ana_tag == '[]':
            # Degenerate
            lemma.append(ana_surface if keep_surface else ana_deep)
            keep_surface = True
        elif ana_tag not in _pos_drop:
            inf.append(ana_tag)

    return ''.join(lemma[::-1]), inf[::-1]


def _get_field_selectors():
    return {
        name: obj for name, obj in inspect.getmembers(sys.modules[__name__])
        if name.startswith('field_') and (
            (inspect.isfunction(obj) and obj.__module__ == __name__) or
            isinstance(obj, itemgetter)
        )
    }


def get_field_function(field):
    """Returns the named field function."""
    return _get_field_selectors()['field_' + field]


def list_field_functions():
    """Lists the available field functions."""
    return [f[6:] for f in _get_field_selectors().keys()]


def convert_token(token, field_fun, lowercase):
    """
    Converts a CoNLL token according to field_fun. If lowercase is true, the
    word or lemma part is lowercased.

    Note: the mapping is one-to-many, as the field functions return a list of
    strings.
    """
    if lowercase:
        token[WORD] = token[WORD].lower()
        token[LEMMA] = token[LEMMA].lower()
    return field_fun(token)
