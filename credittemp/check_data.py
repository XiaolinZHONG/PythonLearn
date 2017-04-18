





def check_ym_data(dataAddress,saveAddress,columnNames):
    import pandas as pd

    reader = pd.read_csv(dataAddress, sep = ',', iterator = True)
    chunks = []
    loop = True
    i = 1
    while loop:
        try:
            chunk = reader.get_chunk(30000)
            chunks.append(chunk)
            print ("正在读取数据块: "+str(i))
            i += 1
        except StopIteration:
            loop = False
            print ('数据读取完成')
    data = pd.concat(chunks, ignore_index = True)
    data=data.fillna(-900)
    # print(data.head())

    data2=pd.DataFrame()
    data2["id"]=data["﻿hb_id"]
    coltmp=[]
    for col in columnNames:
        col2=str(col)+"_b"
        col1=str(col)
        col3=str(col)+"_delta"
        tmp=abs(data[col2] - data[col1])
        if tmp.any()>0.000001:
            data2[col2]=data[col2]
            data2[col1]=data[col1]
            data2[col3]=data[col2]-data[col1]
            coltmp.append(col3)
    # data3=pd.DataFrame()
    # for i in coltmp:
    #     data3=pd.concat([data3,data2.ix[lambda data2: data2[i]> 0.000001,:]],axis=0)
    # print (data2[data2[coltmp]!=0].dropna(how="all"))
    # print(data2.where(data2[coltmp]>0.00001,data2[coltmp],inplace=True))
    # print(data3)
    # data2.to_csv(saveAddress)
    writer=pd.ExcelWriter(saveAddress)
    data2.to_excel(writer,"Sheet1")
    writer.save()
    return(print("任务结束"))


#  "complaint_ratio_ord_12m",
# "highamt_uid_ord_rat2y",
# "htl_star_uid_ord_rat1y",
# "htl_uid_newamt_msx2y",
# "htl_uid_ord_max2y",
# "htl_uidip_ord_ris1y",
# "lowamt_uid_ord_dmnsmax",
# "rcy_band_email",
# "uid_newamt_msx2y",
# "uid_ordtype_cnt2y",
# "train_bu_uid_ord_rat2y",
# "c_uidcid_ord_cnt2y",
# "c_uidip_amt_sum3m",
# "uidcity_ord_ent2y",
# "ns_uidcid_ord_rat2y",
# "ns_uidcid_ord_avg2y",
# "ns_uid_ord_dmnsmax"









if __name__ == '__main__':

    dataAddress="F:/检查数据/other200.csv"
    saveAddress="F:/zxldatacheck20000.xlsx"
    columnNames=[
#  "complaint_ratio_ord_12m",
# "highamt_uid_ord_rat2y",
# "htl_star_uid_ord_rat1y",
# "htl_uid_newamt_msx2y",
# "htl_uid_ord_max2y",
# "htl_uidip_ord_ris1y",
# "lowamt_uid_ord_dmnsmax",
# "rcy_band_email",
# "uid_newamt_msx2y",
# "uid_ordtype_cnt2y",
# "train_bu_uid_ord_rat2y",
# "c_uidcid_ord_cnt2y",
# "c_uidip_amt_sum3m",
# "uidcity_ord_ent2y",
# "ns_uidcid_ord_rat2y",
# "ns_uidcid_ord_avg2y",
# "ns_uid_ord_dmnsmax"]

#     HWH
#     columnNames = ["train_uid_amt_msn",
# "uidcc_ordrat_max2y",
# "flt_uidip_ord_cnt1y",
# "nc_uid_ord_cnt2y",
# "uid_ord_3mcon_cnt",
# "c_uidcid_ord_cnt6m",
# "reg_uidcitylevel_age_z2y",
# "fst_uidcitylevel_age_z2y",
# "payhabitsum_ltr_6_12m",
# "highamt_uid_ord_rat1y",
# "uidip_newamt_sum2y",
# "payhabitsum_countsum_6m",
# "highamt_uid_ord_ris1y"]
    check_ym_data(dataAddress,saveAddress,columnNames)