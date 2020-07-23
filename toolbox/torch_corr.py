import torch

def dissimilarity(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return 1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov_m = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov_m

#source: https://gist.github.com/ncullen93/58e71c4303b89e420bd8e0b0aa54bf48
def cov_2_cor(cov_m):
    d = torch.diag(cov_m)
    stddev = torch.pow(d, 0.5)
    cor_m = cov_m.div(stddev.expand_as(cov_m))
    cor_m = cor_m.div(stddev.expand_as(cor_m).t())
    cor_m = torch.clamp(cor_m, -1.0, 1.0)
    return cor_m

def cor(m):
    #calculate the covariance matrix
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov_m = 1 / (x.size(1) - 1) * x.mm(x.t())
    
    #convert correlation matrix to covariance
    d = torch.diag(cov_m)
    sigma = torch.pow(d, 0.5)
    cor_m = cov_m.div(sigma.expand_as(cov_m))
    cor_m = cor_m.div(sigma.expand_as(cor_m).t())
    cor_m = torch.clamp(cor_m, -1.0, 1.0)
    return cor_m