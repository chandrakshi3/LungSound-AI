def segment_interpretation(start,end,crackles_score,wheezes_score,threshold=.5):
    flags=[]
    if crackles_score>=threshold: flags.append('elevated crackles model score')
    if wheezes_score>=threshold: flags.append('elevated wheezes model score')
    return {'start':float(start),'end':float(end),'interpretation':', '.join(flags) if flags else 'no score crossed threshold','caveat':'Highlights show model output, not a biological cause.'}
